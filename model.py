import torch
from torch import nn
from torchvision import models
from torch.nn import functional as F
import pytorch_lightning as pl


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, should_transfer=False):
        super().__init__()
        self.classifier = models.resnet18(pretrained=should_transfer)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred_y = self(x)
        loss = self.criterion(pred_y, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred_y = self(x)
        loss = self.criterion(pred_y, y)
        self.log('val_loss', loss)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        pred_y = self(x)
        loss = self.criterion(pred_y, y)
        self.log('val_loss', loss)