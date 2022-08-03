import torch
import torchmetrics
import resnet18_handmade as handmade
from torch import nn
from torchvision import models
import pytorch_lightning as pl


class ResNevus(pl.LightningModule):
    def __init__(self, class_balance, im_size=128, should_transfer=False, model_type='simple'):
        super().__init__()
        self.class_balance = class_balance
        self.num_classes = len(class_balance)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = 0
        self.val_acc = 0
        self.test_acc = 0
        if model_type == 'resnet':
            self.classifier = models.resnet18(pretrained=should_transfer)
            linear_size = list(self.classifier.children())[-1].in_features
            self.classifier.fc = nn.Linear(linear_size, self.num_classes)
        elif model_type == 'handmade':
            self.classifier = handmade.ResNet(handmade.BasicBlock, [1,1,1,1], num_classes=self.num_classes)
            linear_size = list(self.classifier.children())[-1].in_features
            self.classifier.fc = nn.Linear(linear_size, self.num_classes)
        elif model_type == 'simple':
            self.classifier = nn.Sequential(
                nn.Conv2d(3,3,9, padding='same'),
                nn.BatchNorm2d(3),
                nn.ReLU(),
                nn.Conv2d(3,3,7, padding='same'),
                nn.BatchNorm2d(3),
                nn.ReLU(),
                nn.Conv2d(3,1,5, padding='same'),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                nn.Conv2d(1,1,3, padding='same'),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(im_size**2, self.num_classes)
            )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        return self.optimizer

    def on_train_epoch_start(self):
        self.train_acc = 0
        self.train_count = 0

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred_y = self(x)
        loss = self.criterion(pred_y, y)
        #conf_mat = torchmetrics.ConfusionMatrix(self.num_classes)
        #conf_mat = conf_mat(torch.argmax(pred_y, 1).to(device='cpu'), torch.argmax(y, 1).to(device='cpu'))
        #for i in range(self.num_classes):
        #    self.train_acc += conf_mat[i, i]/(self.num_classes*self.class_balance[i])
        self.train_acc += torch.sum(torch.argmax(pred_y, 1).to(device='cpu') == torch.argmax(y, 1).to(device='cpu'))
        self.train_count += torch.argmax(pred_y, 1).shape[0]
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        self.train_acc = self.train_acc/self.train_count
        self.log('train_acc', self.train_acc, prog_bar=True)

    def on_validation_epoch_start(self):
        self.val_acc = 0
        self.val_count = 0

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred_y = self(x)
        loss = self.criterion(pred_y, y)
        #conf_mat = torchmetrics.ConfusionMatrix(self.num_classes)
        #conf_mat = conf_mat(torch.argmax(pred_y, 1).to(device='cpu'), torch.argmax(y, 1).to(device='cpu'))
        #for i in range(self.num_classes):
        #    self.val_acc += conf_mat[i, i] / (self.num_classes * self.class_balance[i])
        self.val_acc += torch.sum(torch.argmax(pred_y, 1).to(device='cpu') == torch.argmax(y, 1).to(device='cpu'))
        self.val_count += torch.argmax(pred_y, 1).shape[0]
        self.log("val_loss", loss)

    def validation_epoch_end(self, outputs):
        self.val_acc = self.val_acc/self.val_count
        self.log('val_acc', self.val_acc, prog_bar=True)

    def on_test_epoch_start(self):
        self.test_acc = 0
        self.test_count

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        pred_y = self(x)
        loss = self.criterion(pred_y, y)
        #conf_mat = torchmetrics.ConfusionMatrix(self.num_classes)
        #conf_mat = conf_mat(torch.argmax(pred_y, 1).to(device='cpu'), torch.argmax(y, 1).to(device='cpu'))
        #for i in range(self.num_classes):
        #    self.test_acc += conf_mat[i, i] / (self.num_classes * self.class_balance[i])
        self.test_acc += torch.sum(torch.argmax(pred_y, 1).to(device='cpu') == torch.argmax(y, 1).to(device='cpu'))
        self.test_count += torch.argmax(pred_y, 1).shape[0]
        self.log("test_loss", loss)

    def test_epoch_end(self):
        self.test_acc = self.test_acc/self.val_count
        self.log('test_acc', self.test_acc)
