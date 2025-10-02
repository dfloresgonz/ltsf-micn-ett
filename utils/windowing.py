import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def create_windows(series, input_len=96, output_len=24):
  data = np.asarray(series, dtype=np.float32)
  X, Y = [], []
  L = len(data)
  for i in range(L - input_len - output_len + 1):
    X.append(data[i:i+input_len])
    Y.append(data[i+input_len:i+input_len+output_len])
  return np.stack(X), np.stack(Y)


def split_data(X, Y, train_ratio=0.7, val_ratio=0.1):
  n = len(X)
  n_train = int(n * train_ratio)
  n_val = int(n * val_ratio)
  X_train, Y_train = X[:n_train], Y[:n_train]
  X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
  X_test, Y_test = X[n_train+n_val:], Y[n_train+n_val:]
  return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


class WindowDataset(Dataset):
  def __init__(self, X, Y, mean=None, std=None):
    # normalización (z-score) usando stats del train
    if mean is None:
      mean = X.mean()
    if std is None or std == 0:
      std = X.std() + 1e-6
    self.mean, self.std = float(mean), float(std)
    self.X = (X - self.mean) / self.std
    self.Y = (Y - self.mean) / self.std

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)  # [1, L]
    y = torch.tensor(self.Y[idx], dtype=torch.float32).unsqueeze(0)  # [1, O]
    return x, y


def make_dataloaders(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                     batch_size=32, num_workers=0):
  mean, std = X_train.mean(), X_train.std()
  ds_train = WindowDataset(X_train, Y_train, mean, std)
  ds_val = WindowDataset(X_val,   Y_val,   mean, std)
  ds_test = WindowDataset(X_test,  Y_test,  mean, std)
  return (
      DataLoader(ds_train, batch_size=batch_size,
                 shuffle=True,  num_workers=num_workers),
      DataLoader(ds_val,   batch_size=batch_size,
                 shuffle=False, num_workers=num_workers),
      DataLoader(ds_test,  batch_size=batch_size,
                 shuffle=False, num_workers=num_workers),
      mean, std
  )
