from einops import rearrange
from attention import Attention
from mlp import MLP
from drop_path import DropPath
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, emb_dim=128, heads=5, p=0.5, phase='train'):
        super().__init__()
        self.att = Attention(emb_dim, heads)
        self.ln = nn.LayerNorm(emb_dim)
        self.mlp = MLP(emb_dim)
        self.drop_path = DropPath(p, phase)
        self.temporal_fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, P, T):
        # # B P*T+1 D | 2 1569 128
        # time attention
        init_cls_token = x[:, 0, :].unsqueeze(1)
        x = x[:, 1:, :]
        xt = rearrange(x, 'b (t p) d -> (b p) t d', p=P, t=T)  # B P*T D | 392 8 128
        xt = self.drop_path(self.att(self.ln(xt)))  # B P*T D | 392 8 128
        xt = rearrange(xt, '(b p) t m -> b (p t) m', t=T, p=P)  # B p*T D | [2, 1568, 128]
        xt = xt + x  # B P*T D  | 392 8 128

        # space attention
        xs = xt
        xs = rearrange(xs, 'b (p t) d -> (b t) p d', p=P, t=T)  # B*T P D  | 16 196 128
        cls_token = init_cls_token.repeat(1, T, 1)  # B T D  | 2 8 128
        cls_token = rearrange(cls_token, 'b t d -> (b t) d').unsqueeze(1)  # B*T 1 D  | 16 1 128
        xs = torch.cat((cls_token, xs), dim=1)  # B*T P+1 D  | 16 197 128
        xs = self.drop_path(self.att(self.ln(xs)))  # B*T P D  | 16 197 128

        # processing cls_token
        cls_token = xs[:, 0, :]  # 16 128
        cls_token = rearrange(cls_token, '(b t) m -> b t m', t=T)  # B T D  | 2 8 128
        cls_token = torch.mean(cls_token, 1, True)  # B 1 D  | 2 1 128

        xs = xs[:, 1:, :]
        xs = rearrange(xs, '(b t) p d -> b (t p) d', t=T, p=P)  # B T*P D  | 2 1568 128

        # B T*P D | 2 1569 128
        x_mlp = torch.cat((init_cls_token, xt), dim=1) + torch.cat((cls_token, xs), dim=1)
        x = self.drop_path(self.mlp(self.ln(x_mlp)))  # B T*P D | 2 1569 128
        return x_mlp + x  # B T*P D | 2 1569 128

