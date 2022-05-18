import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(3, 3, 6, padding="same"),
            nn.MaxPool2d(5),
            nn.Conv2d(60, 1, 5, padding="same"),
            nn.MaxPool2d(5),
            nn.Flatten(),
            nn.Linear(40*40, 3)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        pred_y = self.classifier(x)
        loss = F.mse_loss(pred_y, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        pred_y = self.encoder(x)
        loss = F.mse_loss(pred_y, x)
        self.log('val_loss', loss)
