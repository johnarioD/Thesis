import torch
import torchmetrics
import resnet18_handmade as handmade
from torch import nn
from torchvision import models
import pytorch_lightning as pl
import contextlib
import torch.nn.functional as F


class BaselineModel(pl.LightningModule):
    def __init__(self, class_balance, im_size=512, should_transfer=False, model_type='simple'):
        super().__init__()
        self.class_balance = class_balance
        self.num_classes = len(class_balance)

        self.criterion = nn.CrossEntropyLoss()
        self.acc_metric = torchmetrics.Accuracy()
        self.accuracy = {"train":0,"test":0,"val":0}

        if model_type == 'resnet18':
            self.classifier = models.resnet18(pretrained=should_transfer)
            linear_size = list(self.classifier.children())[-1].in_features
            self.classifier.fc = nn.Linear(linear_size, self.num_classes)
        elif model_type == 'resnet18_handmade':
            self.classifier = handmade.ResNet(handmade.BasicBlock, [1, 1, 1, 1], num_classes=self.num_classes)
            linear_size = list(self.classifier.children())[-1].in_features
            self.classifier.fc = nn.Linear(linear_size, self.num_classes)
        elif model_type == 'simple_conv':
            self.classifier = nn.Sequential(
                nn.Conv2d(3, 3, 9, padding='same'),
                nn.BatchNorm2d(3),
                nn.ReLU(),
                nn.Conv2d(3, 3, 7, padding='same'),
                nn.BatchNorm2d(3),
                nn.ReLU(),
                nn.Conv2d(3, 1, 5, padding='same'),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                nn.Conv2d(1, 1, 3, padding='same'),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(im_size ** 2, self.num_classes)
            )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        return self.optimizer

    def generic_step(self, train_batch, batch_idx, step_type):
        x, y = train_batch
        pred_y = self(x)
        loss = self.criterion(pred_y, y)
        self.accuracy[step_type] = self.acc_metric(torch.argmax(pred_y, 1), y)
        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self.generic_step(train_batch, batch_idx, step_type="train")
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.accuracy['train'], prog_bar=True)

    def validation_step(self, val_batch, batch_idx):
        loss = self.generic_step(val_batch, batch_idx, step_type="val")
        self.log("val_loss", loss)

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.accuracy['val'], prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        loss = self.generic_step(test_batch, batch_idx, step_type="test")
        self.log("test_loss", loss)

    def test_epoch_end(self):
        self.log('test_acc', self.accuracy['test'])


class VATModel(pl.LightningModule):
    def __init__(self, class_balance, xi=10.0, eps=1.0, ip=1, a=1, should_transfer=False):
        super().__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.a = a

        self.criterion = nn.CrossEntropyLoss()
        self.acc_metric = torchmetrics.Accuracy()

        self.accuracy = {"train":0,"test":0,"val":0}

        self.class_balance = class_balance
        self.num_classes = len(class_balance)

        self.classifier = models.resnet18(pretrained=should_transfer)
        linear_size = list(self.classifier.children())[-1].in_features
        self.classifier.fc = nn.Linear(linear_size, self.num_classes)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        self.classifier(x)

    def configure_optimizers(self):
        return self.optimizer

    def generic_step(self, train_batch, batch_idx, step_type):
        x, y = train_batch
        unlabeled_idx = y is None

        # Create random unit tensor
        d = torch.rand(x.shape).to(x.device)
        d = d/(torch.norm(d) + 1e-8)

        pred_y = self.classifier(x)
        y[unlabeled_idx] = pred_y[unlabeled_idx]
        l = self.criterion(pred_y, y)
        # with torch.no_grad():
        # pred = F.softmax(self.classifier(x), dim=1)

        R_adv = torch.zeros_like(x)
        for _ in range(self.ip):
            r = self.xi * d
            r.requires_grad = True
            pred_hat = self.classifier(x + r)
            # pred_hat = F.log_softmax(pred_hat, dim=1)
            D = self.criterion(pred_hat, pred_y)
            self.classifier.zero_grad()
            D.requires_grad=True
            D.backward()
            R_adv += self.eps * r.grad / (torch.norm(r.grad) + 1e-8)
        R_adv /= 32
        loss = l + R_adv * self.a
        loss.backward()
        self.accuracy[step_type] = self.acc_metric(torch.argmax(pred_y, 1), y)
        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self.generic_step(train_batch, batch_idx, step_type="train")
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.accuracy['train'], prog_bar=True)

    def validation_step(self, val_batch, batch_idx):
        loss = self.generic_step(val_batch, batch_idx, step_type="val")
        self.log("val_loss", loss)

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.accuracy['val'], prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        loss = self.generic_step(test_batch, batch_idx, step_type="test")
        self.log("test_loss", loss)

    def test_epoch_end(self):
        self.log('test_acc', self.accuracy['test'])
