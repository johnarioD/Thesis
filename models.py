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
        self.train_acc = 0
        self.test_acc = 0
        self.val_acc = 0
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

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred_y = self(x)
        loss = self.criterion(pred_y, y)
        self.train_acc = self.acc_metric(torch.argmax(pred_y, 1), y)
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.train_acc, prog_bar=True)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred_y = self(x)
        loss = self.criterion(pred_y, y)
        self.val_acc = self.acc_metric(torch.argmax(pred_y, 1), y)
        self.log("val_loss", loss)

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_acc, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        pred_y = self(x)
        loss = self.criterion(pred_y, y)
        self.test_acc = self.acc_metric(torch.argmax(pred_y, 1), y)
        self.log("test_loss", loss)

    def test_epoch_end(self):
        self.log('test_acc', self.test_acc)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATModel(pl.LightningModule):
    def __init__(self, class_balance, should_transfer=False, xi=10.0, eps=1.0, ip=1, a=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super().__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.a = a

        self.cross_entropy = nn.CrossEntropyLoss()
        self.acc_metric = torchmetrics.Accuracy()

        self.train_acc = 0
        self.test_acc = 0
        self.val_acc = 0

        self.class_balance = class_balance
        self.num_classes = len(class_balance)
        self.classifier = models.resnet18(pretrained=should_transfer)
        linear_size = list(self.classifier.children())[-1].in_features
        self.classifier.fc = nn.Linear(linear_size, self.num_classes)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        self.classifier(x)

    def calc_Radv(self, x):
        with torch.no_grad():
            pred = F.softmax(self.classifier(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        # calc adversarial direction
        R_adv = 0
        for _ in range(self.ip):
            d.requires_grad_()
            r = self.xi * d
            pred_hat = self.classifier(x + r)
            pred_hat = F.log_softmax(pred_hat, dim=1)
            loss = self.cross_entropy(pred, pred_hat)
            loss.backward()
            R_adv += loss.grad/_l2_normalize(loss.grad)*self.eps
            self.classifier.zero_grad()

        return R_adv

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        Radv = self.VAT_loss(self.classifier, x)
        pred_y = self(x)
        none_idx = y is None
        y[none_idx] = pred_y[none_idx]
        loss = self.criterion(pred_y, y) + Radv * self.VAT_loss.a
        loss.backward()
        self.train_acc = self.acc_metric(torch.argmax(pred_y, 1), y)
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.train_acc, prog_bar=True)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        Radv = self.VAT_loss(self.classifier, x)
        pred_y = self(x)
        none_idx = y == None
        y[none_idx] = pred_y[none_idx]
        loss = self.criterion(pred_y, y) + Radv * self.VAT_loss.a
        loss.backward()
        self.val_acc = self.acc_metric(torch.argmax(pred_y, 1), y)
        self.log("val_loss", loss)

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_acc, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        Radv = self.VAT_loss(self.classifier, x)
        pred_y = self(x)
        none_idx = y is None
        y[none_idx] = pred_y[none_idx]
        loss = self.criterion(pred_y, y) + Radv * self.VAT_loss.a
        loss.backward()
        self.test_acc = self.acc_metric(torch.argmax(pred_y, 1), y)
        self.log("test_loss", loss)

    def test_epoch_end(self):
        self.log('test_acc', self.test_acc)