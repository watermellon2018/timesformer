from embedding import EmbeddingPatch
from encoder import Encoder
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class TimeSFormer(pl.LightningModule):
    def __init__(self, num_class=3, emb_dim=128, patch_size=16, img_size=224, count_frame=8,
                 count_encoders=2, count_heads=5, p=0.5, phase='train'):
        super().__init__()
        self.patch_emb = EmbeddingPatch(emb_dim, patch_size, img_size)
        self.count_patch = self.patch_emb.count_patch

        self.space_pos = nn.Parameter(torch.zeros((1, self.count_patch, emb_dim)))
        self.time_pos = nn.Parameter(torch.zeros((1, count_frame, emb_dim)))
        self.cls_token = nn.Parameter(torch.zeros((1, emb_dim)))

        self.count_encoders = count_encoders
        self.count_heads = count_heads
        self.block = nn.ModuleList([
            Encoder(emb_dim, count_heads, p, phase)
            for i in range(self.count_encoders)
        ])

        self.head = nn.Linear(emb_dim, num_class)

    def forward(self, img):
        B, T, C, H, W = img.shape

        emb = self.patch_emb(img)  # B*T P D | 16 196 128
        emb += self.space_pos

        # add time position
        emb = rearrange(emb, '(b t) p d -> (b p) t d', p=self.count_patch, t=T)  # B*P T D
        emb += self.time_pos
        emb = rearrange(emb, '(b p) t d -> b (p t) d', p=self.count_patch, t=T)  # B P*T D | 2 1568 128

        # add classification token
        cls_token = self.cls_token.expand(B, -1, -1)  # B 1 D  | 2 1 128
        x = torch.cat((cls_token, emb), dim=1)  # B P*T+1 D | 2 1569 128

        # encoder
        for blk in self.block:
            x = blk(x, self.count_patch, T)  # B T*P D | [2, 1569, 128]
        y = self.head(x)

        return y[:, 0, :]  # cls_token

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        # print(y_pred.shape, y.shape) # [2 3] # 2
        loss = F.cross_entropy(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer