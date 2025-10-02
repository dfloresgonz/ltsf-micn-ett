import torch
import torch.nn as nn


class TrendRegression(nn.Module):
  """
  Predice la tendencia Xt hacia el futuro con una proyección lineal temporal:
  - Entrada: Xt [B, 1, L]
  - Salida:  Ytrend [B, 1, O]
  Implementa MICN-regre con nn.Linear(L, O).
  """

  def __init__(self, input_len: int, output_len: int):
    super().__init__()
    self.input_len = input_len
    self.output_len = output_len
    self.head = nn.Linear(input_len, output_len, bias=True)

  def forward(self, Xt):
    B, C, L = Xt.shape
    assert C == 1 and L == self.input_len
    y = self.head(Xt.squeeze(1))
    return y.unsqueeze(1)
