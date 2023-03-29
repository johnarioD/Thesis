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

        self.num_steps = {"train":0, "val":0, "test":0}
        self.cum_loss = {"train":0, "val":0, "test":0}

        self.accuracy = {"train": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted'),
                         "test": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted'),
                         "val": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted')}
                
        self.classifier = models.resnet18(pretrained=(pretrained==1))
        if self.num_classes==2:
            self.auc = {"train": torchmetrics.AUROC(num_classes=num_classes, pos_label=1),
                        "val": torchmetrics.AUROC(num_classes=num_classes, pos_label=1),
                        "test": torchmetrics.AUROC(num_classes=num_classes, pos_label=1)}
            self.confmat = {"train": torchmetrics.ConfusionMatrix(task="binary", num_classes=2),
                        "val": torchmetrics.ConfusionMatrix(task="binary", num_classes=2),
                        "test": torchmetrics.ConfusionMatrix(task="binary", num_classes=2)}
        if self.num_classes==3:
            self.confmat = {"train": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes),
                            "val": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes),
                            "test": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)}
            
        linear_size = list(self.classifier.children())[-1].in_features
        self.classifier.fc = nn.Linear(linear_size, self.num_classes)

    def change_dims(self, num_classes):
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
        if self.num_classes==3:
            self.confmat = {"train": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes),
                            "val": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes),
                            "test": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)}
            
        linear_size = list(self.classifier.children())[-1].in_features
        self.classifier.fc = nn.Linear(linear_size, self.num_classes)

    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_epoch_start(self):
        self.cum_loss['train']=0
        self.num_steps['train']=0

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred_y = self.softmax(self(x))
        loss = self.cross_entropy(pred_y, y)
        self.accuracy['train'].update(torch.argmax(pred_y, 1).to('cpu'), y.to('cpu'))
        if self.num_classes == 2:
            self.auc['train'].update(self.softmax(pred_y)[:, 1], y)
        self.num_steps['train']+=1
        self.cum_loss['train']+=loss
        return {'loss': loss, 'preds': pred_y, "target": y}

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
        x, y = val_batch
        pred_y = self.softmax(self(x))
        loss = self.cross_entropy(pred_y, y)
        self.accuracy['val'].update(torch.argmax(pred_y, 1).to('cpu'), y.to('cpu'))
        if self.num_classes == 2:
            self.auc['val'].update(self.softmax(pred_y)[:, 1], y)
        self.num_steps['val']+=1
        self.cum_loss['val']+=loss
        return {'loss': loss, 'preds': pred_y, "target": y}

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
        x, y = test_batch
        pred_y = self.softmax(self(x))
        loss = self.cross_entropy(pred_y, y)
        self.accuracy['test'].update(torch.argmax(pred_y, 1).to('cpu'), y.to('cpu'))
        if self.num_classes == 2:
            self.auc['test'].update(self.softmax(pred_y)[:, 1], y)
        self.confmat["test"].update(torch.argmax(pred_y, 1).to('cpu'), y.to('cpu'))
        self.num_steps['test']+=1
        self.cum_loss['test']+=loss
        return {'loss': loss, 'preds': pred_y, "target": y}

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

class VotingModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.softmax = torch.nn.Softmax(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss()

        self.num_steps = {"train":0, "val":0, "test":0}
        self.cum_loss = {"train":0, "val":0, "test":0}

        self.accuracy = {"train": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted'),
                         "test": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted'),
                         "val": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted')}
                
        self.classifier_0 = models.resnet18(pretrained=False)
        self.classifier_1 = models.resnet18(pretrained=True)
        tmp = BaselineModel.load_from_checkpoint(checkpoint_path="./models/baseline_ISIC_1.chkpt", num_classes=2)
        tmp.change_dims(num_classes=num_classes)
        self.classifier_2 = tmp.classifier
        if self.num_classes==2:
            self.auc = {"train": torchmetrics.AUROC(num_classes=num_classes, pos_label=1),
                        "val": torchmetrics.AUROC(num_classes=num_classes, pos_label=1),
                        "test": torchmetrics.AUROC(num_classes=num_classes, pos_label=1)}
            self.confmat = {"train": torchmetrics.ConfusionMatrix(task="binary", num_classes=2),
                        "val": torchmetrics.ConfusionMatrix(task="binary", num_classes=2),
                        "test": torchmetrics.ConfusionMatrix(task="binary", num_classes=2)}
        if self.num_classes==3:
            self.confmat = {"train": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes),
                            "val": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes),
                            "test": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)}
            
        linear_size = list(self.classifier_0.children())[-1].in_features
        self.classifier_0.fc = nn.Linear(linear_size, self.num_classes)
        linear_size = list(self.classifier_1.children())[-1].in_features
        self.classifier_1.fc = nn.Linear(linear_size, self.num_classes)
        linear_size = list(self.classifier_2.children())[-1].in_features
        self.classifier_2.fc = nn.Linear(linear_size, self.num_classes)

    def forward(self, x):
        return self.classifier_0(x) + self.classifier_1(x) + self.classifier_2(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_epoch_start(self):
        self.cum_loss['train']=0
        self.num_steps['train']=0

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred_y = self.softmax(self(x))
        loss = self.cross_entropy(pred_y, y)
        self.accuracy['train'].update(torch.argmax(pred_y, 1).to('cpu'), y.to('cpu'))
        if self.num_classes == 2:
            self.auc['train'].update(self.softmax(pred_y)[:, 1], y)
        self.num_steps['train']+=1
        self.cum_loss['train']+=loss
        return {'loss': loss, 'preds': pred_y, "target": y}

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
        x, y = val_batch
        pred_y = self.softmax(self(x))
        loss = self.cross_entropy(pred_y, y)
        self.accuracy['val'].update(torch.argmax(pred_y, 1).to('cpu'), y.to('cpu'))
        if self.num_classes == 2:
            self.auc['val'].update(self.softmax(pred_y)[:, 1], y)
        self.num_steps['val']+=1
        self.cum_loss['val']+=loss
        return {'loss': loss, 'preds': pred_y, "target": y}

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
        x, y = test_batch
        pred_y = self.softmax(self(x))
        loss = self.cross_entropy(pred_y, y)
        self.accuracy['test'].update(torch.argmax(pred_y, 1).to('cpu'), y.to('cpu'))
        if self.num_classes == 2:
            self.auc['test'].update(self.softmax(pred_y)[:, 1], y)
        self.confmat["test"].update(torch.argmax(pred_y, 1).to('cpu'), y.to('cpu'))
        self.num_steps['test']+=1
        self.cum_loss['test']+=loss
        return {'loss': loss, 'preds': pred_y, "target": y}

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
