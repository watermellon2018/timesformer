from math import sqrt
import torch
import torch.nn as nn

class Attention(nn.Module):
  def __init__(self, emb_dim=128, heads=5):
    super().__init__()
    self.w_q = nn.Linear(emb_dim, emb_dim)
    self.w_k = nn.Linear(emb_dim, emb_dim)
    self.w_v = nn.Linear(emb_dim, emb_dim)
    self.w_0 = nn.Linear(emb_dim, emb_dim)

    self.sm = nn.Softmax()
    self.heads = heads
    self.dim_head = emb_dim / heads

  def forward(self, x):
    q = self.w_q(x) # 392 8 128
    k = self.w_k(x)
    v = self.w_v(x)
    z = q @ torch.transpose(k, -2, -1) # 392 8 8

    att = self.sm(z / sqrt(self.dim_head))
    att = att @ v
    att = self.w_0(att)

    return att + x