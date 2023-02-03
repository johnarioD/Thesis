import torch
import torchmetrics
import resnet18_handmade as handmade
from torch import nn
from torchvision import models
import pytorch_lightning as pl
import contextlib
from torch.nn import Softmax
import torch.nn.functional as F


class BaselineModel(pl.LightningModule):
    def __init__(self, num_classes, pretrained=0):
        super().__init__()
        self.num_classes = num_classes

        self.softmax = torch.nn.Softmax(dim=1)

        self.cross_entropy = nn.CrossEntropyLoss()

        self.num_steps = {"train":0,
                        "val":0,
                        "test":0}
        self.cum_loss = {"train":0,
                        "val":0,
                        "test":0}

        self.accuracy = {"train": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted'),
                         "test": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted'),
                         "val": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted')}
        if self.num_classes==2:
            self.auc = {"train": torchmetrics.AUROC(num_classes=num_classes, pos_label=1),
                        "val": torchmetrics.AUROC(num_classes=num_classes, pos_label=1),
                        "test": torchmetrics.AUROC(num_classes=num_classes, pos_label=1)}
            self.confmat = {"train": torchmetrics.ConfusionMatrix(task="binary", num_classes=2),
                        "val": torchmetrics.ConfusionMatrix(task="binary", num_classes=2),
                        "test": torchmetrics.ConfusionMatrix(task="binary", num_classes=2)}

        self.confmat = {"train": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes),
                        "val": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes),
                        "test": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)}

        self.classifier = models.resnet18(pretrained=pretrained==1)

        linear_size = list(self.classifier.children())[-1].in_features
        self.classifier.fc = nn.Linear(linear_size, self.num_classes)

    def change_output(self, num_classes=2):
        self.num_classes = num_classes

        self.accuracy = {"train": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted'),
                         "test": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted'),
                         "val": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted')}
        if self.num_classes==2:
            self.auc = {"train": torchmetrics.AUROC(num_classes=num_classes, pos_label=1),
                        "val": torchmetrics.AUROC(num_classes=num_classes, pos_label=1),
                        "test": torchmetrics.AUROC(num_classes=num_classes, pos_label=1)}
            self.confmat = {"train": torchmetrics.ConfusionMatrix(task="binary", num_classes=2),
                        "val": torchmetrics.ConfusionMatrix(task="binary", num_classes=2),
                        "test": torchmetrics.ConfusionMatrix(task="binary", num_classes=2)}

        self.confmat = {"train": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes),
                        "val": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes),
                        "test": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)}

        linear_size = list(self.classifier.children())[-1].in_features
        self.classifier.fc = nn.Linear(linear_size, self.num_classes)

    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def generic_step(self, train_batch, batch_idx, step_type):
        x, y = train_batch
        pred_y = self.softmax(self(x))
        loss = self.cross_entropy(pred_y, y)
        self.accuracy[step_type].update(torch.argmax(pred_y, 1).to('cpu'), y.to('cpu'))
        if self.num_classes == 2:
            self.auc[step_type].update(self.softmax(pred_y)[:, 1], y)
        if step_type=="test":
            self.confmat["test"].update(torch.argmax(pred_y, 1).to('cpu'), y.to('cpu'))
        self.num_steps[step_type]+=1
        self.cum_loss[step_type]+=loss
        return {'loss': loss, 'preds': pred_y, "target": y}

    def training_epoch_start(self):
        self.cum_loss['train']=0
        self.num_steps['train']=0

    def training_step(self, train_batch, batch_idx):
        out = self.generic_step(train_batch, batch_idx, step_type="train")
        return out

    def training_epoch_end(self, outputs):
        self.cum_loss['train'] /= self.num_steps['train']
        accuracy = self.accuracy['train'].compute()
        self.log("train_loss", self.cum_loss['train'])
        self.log('train_acc', accuracy, prog_bar=True)
        if self.num_classes == 2:
            auc = self.auc['train'].compute()
            self.log('train_auc', auc, prog_bar=True)

    def validation_epoch_start(self):
        self.cum_loss['val']=0
        self.num_steps['val']=0

    def validation_step(self, val_batch, batch_idx):
        out = self.generic_step(val_batch, batch_idx, step_type="val")
        return out

    def validation_epoch_end(self, outputs):
        self.cum_loss['val'] /= self.num_steps['val']
        accuracy = self.accuracy['val'].compute()
        self.log("val_loss", self.cum_loss['val'], prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        if self.num_classes == 2:
            auc = self.auc['val'].compute()
            self.log('val_auc', auc)

    def test_epoch_start(self):
        self.cum_loss['test']=0
        self.num_steps['test']=0

    def test_step(self, test_batch, batch_idx):
        out = self.generic_step(test_batch, batch_idx, step_type="test")
        self.log("test_loss", out['loss'])
        return out

    def test_epoch_end(self, outputs):
        self.cum_loss['test'] /= self.num_steps['test']
        accuracy = self.accuracy['test'].compute()
        confmat = self.confmat["test"].compute()
        self.log("test_loss", self.cum_loss['test'])
        self.log('test_acc', accuracy)
        self.log("test_confmat", confmat)
        if self.num_classes == 2:        
            auc = self.auc['test'].compute()
            self.log('test_auc', auc)
