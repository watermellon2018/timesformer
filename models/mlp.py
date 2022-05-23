import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.hidden_feater = nn.Linear(emb_dim, emb_dim)
    self.out_featers = nn.Linear(emb_dim, emb_dim)
    self.gelu = nn.GELU()

  def forward(self, x):
    hidden = self.gelu(self.hidden_feater(x))
    out = self.gelu(self.out_featers(hidden))
    return out
