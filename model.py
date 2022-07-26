import torch
import torchmetrics
from torch import nn
from torchvision import models
import pytorch_lightning as pl


class ResNevus(pl.LightningModule):
    def __init__(self, class_balance, should_transfer=False):
        super().__init__()
        self.classifier = models.resnet18(pretrained=should_transfer)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.class_balance = class_balance
        self.num_classes = len(class_balance)
        self.criterion = nn.CrossEntropyLoss()
        linear_size = list(self.classifier.children())[-1].in_features
        self.classifier.fc = nn.Linear(linear_size, self.num_classes)
        self.train_acc = 0
        self.val_acc = 0
        self.test_acc = 0

    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def on_train_epoch_start(self):
        self.train_acc = 0

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred_y = self(x)
        loss = self.criterion(pred_y, y)
        conf_mat = torchmetrics.ConfusionMatrix(self.num_classes)
        conf_mat = conf_mat(torch.argmax(pred_y, 1).to(device='cpu'), torch.argmax(y, 1).to(device='cpu'))
        for i in range(self.num_classes):
            self.train_acc += conf_mat[i, i]/(self.num_classes*self.class_balance[i])
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.train_acc)


    def on_validation_epoch_start(self):
        self.val_acc = 0

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred_y = self(x)
        loss = self.criterion(y, pred_y)
        conf_mat = torchmetrics.ConfusionMatrix(self.num_classes)
        conf_mat = conf_mat(torch.argmax(pred_y, 1).to(device='cpu'), torch.argmax(y, 1).to(device='cpu'))
        for i in range(self.num_classes):
            self.val_acc += conf_mat[i, i] / (self.num_classes * self.class_balance[i])
        self.log("val_loss", loss)

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_acc)

    def on_test_epoch_start(self):
        self.test_acc = 0

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        pred_y = self(x)
        loss = self.criterion(pred_y, y)
        conf_mat = torchmetrics.ConfusionMatrix(self.num_classes)
        conf_mat = conf_mat(torch.argmax(pred_y, 1).to(device='cpu'), torch.argmax(y, 1).to(device='cpu'))
        for i in range(self.num_classes):
            self.test_acc += conf_mat[i, i] / (self.num_classes * self.class_balance[i])
        self.log("test_loss", loss)

    def test_epoch_end(self):
        self.log('test_acc', self.test_acc)
