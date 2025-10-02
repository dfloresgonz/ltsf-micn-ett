import os
import mlflow
import torch
import torch.nn as nn
import pandas as pd
from micn import MICNModel
from utils.windowing import create_windows, split_data, make_dataloaders
from utils.metrics import mse, mae

mlflow.set_tracking_uri("http://127.0.0.1:5000")


def get_device():
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_eval_model(params):
  """
  params:
    dataset: ETTh1|ETTh2|ETTm1|ETTm2
    input_len: 96
    output_len: 24|48|96|192|336|720
    d_model: 64 (ej.)
    n_layers: 1-2
    scales: (12,24,48)
    batch_size, epochs, lr
  """
  device = get_device()

  # 1) Carga y ventanas
  df = pd.read_csv(os.path.join("../data", f"{params['dataset']}.csv"))
  series = df["OT"].values.astype("float32")

  X, Y = create_windows(
      series, input_len=params["input_len"], output_len=params["output_len"])
  (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_data(X, Y)

  train_loader, val_loader, test_loader, mean, std = make_dataloaders(
      X_train, Y_train, X_val, Y_val, X_test, Y_test,
      batch_size=params["batch_size"]
  )

  # 2) Modelo
  model = MICNModel(
      input_len=params["input_len"],
      output_len=params["output_len"],
      d_model=params["d_model"],
      n_layers=params["n_layers"],
      scales=tuple(params["scales"])
  ).to(device)

  # 3) Optimizador y pérdida
  optimizer = torch.optim.Adam(model.parameters(
  ), lr=params["learning_rate"], weight_decay=params.get("weight_decay", 0.0))
  loss_fn = nn.MSELoss()

  # 4) Entrenamiento
  with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_param("norm_mean", float(mean))
    mlflow.log_param("norm_std",  float(std))

    best_val = float("inf")
    for epoch in range(params["epochs"]):
      model.train()
      train_losses = []

      for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)  # xb:[B,1,L], yb:[B,1,O]
        optimizer.zero_grad()
        yhat, _, _ = model(xb)                # [B,1,O]
        loss = loss_fn(yhat, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())

      # validación
      model.eval()
      val_losses = []
      with torch.no_grad():
        for xb, yb in val_loader:
          xb, yb = xb.to(device), yb.to(device)
          yhat, _, _ = model(xb)
          loss = loss_fn(yhat, yb)
          val_losses.append(loss.item())

      tr_mse = sum(train_losses) / max(1, len(train_losses))
      va_mse = sum(val_losses) / max(1, len(val_losses))

      mlflow.log_metric("train_mse", tr_mse, step=epoch)
      mlflow.log_metric("val_mse",   va_mse, step=epoch)
      print(f"[{params['dataset']} | H={params['output_len']}] "
            f"Epoch {epoch+1}/{params['epochs']}  Train MSE={tr_mse:.4f}  Val MSE={va_mse:.4f}")

      if va_mse < best_val:
        best_val = va_mse
        # guardar mejor estado (opcional)
        torch.save(model.state_dict(), "best.pt")
        mlflow.log_artifact("best.pt")

    # 5) Test
    model.eval()
    test_losses_mse, test_losses_mae = [], []
    with torch.no_grad():
      for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        yhat, _, _ = model(xb)
        test_losses_mse.append(mse(yb, yhat).item())
        test_losses_mae.append(mae(yb, yhat).item())

    test_mse = sum(test_losses_mse) / max(1, len(test_losses_mse))
    test_mae = sum(test_losses_mae) / max(1, len(test_losses_mae))
    mlflow.log_metric("test_mse", test_mse)
    mlflow.log_metric("test_mae", test_mae)
    print(f"TEST  MSE={test_mse:.4f}  MAE={test_mae:.4f}")
