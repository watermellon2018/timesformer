import torch
import torch.nn as nn
from einops import rearrange

class EmbeddingPatch(nn.Module):
    def __init__(self, emb_dim=128, patch_size=16, img_size=224):
        super().__init__()
        self.to_patch = nn.Conv2d(3, emb_dim, patch_size, stride=patch_size)
        self.count_patch = img_size ** 2 // patch_size ** 2

    def forward(self, img):
        img = rearrange(img, 'b t c h w -> (b t) c h w')
        pathces = self.to_patch(img)
        emb = pathces.flatten(2)
        emb = torch.permute(emb, (0, 2, 1))
        return emb