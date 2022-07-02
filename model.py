import torch
from torch import nn
from torchvision import models
import pytorch_lightning as pl


class ResNevus(pl.LightningModule):
    def __init__(self, num_classes, should_transfer=False):
        super().__init__()
        self.classifier = models.resnet18(pretrained=should_transfer)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        linear_size = list(self.classifier.children())[-1].in_features
        self.classifier.fc = nn.Linear(linear_size, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred_y = self(x)
        loss = self.criterion(pred_y, y)
        acc = (torch.argmax(pred_y, 1) == torch.argmax(y, 1)).type(torch.FloatTensor).mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred_y = self(x)
        loss = self.criterion(y, pred_y)
        acc = (torch.argmax(pred_y, 1) == torch.argmax(y, 1)).type(torch.FloatTensor).mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        pred_y = self(x)
        loss = self.criterion(pred_y, y)
        acc = (torch.argmax(pred_y, 1) == torch.argmax(y, 1)).type(torch.FloatTensor).mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
