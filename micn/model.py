import torch.nn as nn
from .decomposition import MHDecomp
from .regression import TrendRegression
from .mic_layers import MICLayers


class MICNModel(nn.Module):
  """
  Ensamble:
  - MHDecomp → Xt, Xs
  - TrendRegression(Xt) → Ytrend
  - MICLayers(Xs) → Yseasonal
  - Ypred = Ytrend + Yseasonal
  """

  def __init__(self, input_len=96, output_len=24,
               d_model=64, n_layers=1, scales=(12, 24, 48)):
    super().__init__()
    self.input_len = input_len
    self.output_len = output_len

    self.decomp = MHDecomp(kernel_sizes=scales)
    self.trend = TrendRegression(input_len=input_len, output_len=output_len)
    self.mic = MICLayers(input_len=input_len, output_len=output_len,
                         d_model=d_model, n_layers=n_layers, scales=scales)

  def forward(self, X):  # X: [B, 1, L]
    Xt, Xs = self.decomp(X)                  # [B,1,L], [B,1,L]
    Ytrend = self.trend(Xt)                  # [B,1,O]
    Yseasonal = self.mic(Xs)                 # [B,1,O]
    return Ytrend + Yseasonal, Ytrend, Yseasonal
