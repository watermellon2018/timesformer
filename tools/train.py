from models.timesformer import TimeSFormer

import torch
import torch.nn as nn
import pytorch_lightning as pl

config = {
    'num_class': 3, # len(classes_to_label),
    'emb_dim': 128,
    'patch_size': 16,
    'img_size': 224,
    'clip_size': 5,
    'count_encoders': 2,
    'count_heads': 5,
}

def train(train_loader, val_loader, config):
    model = TimeSFormer(num_class=config['num_class'], emb_dim=config['emb_dim'],
                        patch_size=config['patch_size'], img_size=config['img_size'],
                        count_frame=config['clip_size'], count_encoders=config['count_encoders'],
                        count_heads=config['count_encoders'], p=0.5, phase='train')

    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model, train_loader, val_loader)
