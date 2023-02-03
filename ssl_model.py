import torch
import torchmetrics
import resnet18_handmade as handmade
from torch import nn
from torchvision import models
import pytorch_lightning as pl
import contextlib
from torch.nn import Softmax
import torch.nn.functional as F

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def kl_div_with_logit(q_logit, p_logit, average=True):
    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    if average is True:
        qlogq = (q * logq).sum(dim=1).mean(dim=0)
        qlogp = (q * logp).sum(dim=1).mean(dim=0)
    else:
        qlogq = (q * logq).sum(dim=1, keepdim=True)
        qlogp = (q * logp).sum(dim=1, keepdim=True)

    return qlogq - qlogp


def l2_normalize(d):
    d_reshaped = d.view(d.size(0), -1)
    d_norm = F.normalize(d_reshaped, dim=1, p=2).view(d.size())
    return d_norm


class VATModel(pl.LightningModule):
    def __init__(self, xi=1e-6, eps=1.0, ip=1, a=1, num_classes=2, pretrained=0):
        super().__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.a = a

        self.num_classes = num_classes
        self.automatic_optimization = False

        self.softmax =  Softmax(dim=1)

        self.kl_div = kl_div_with_logit
        self.cross_entropy = nn.CrossEntropyLoss()
        self.num_steps = {"train":0,
                        "val":0,
                        "test":0}
        self.cum_loss = {"train":0,
                        "val":0,
                        "test":0}
        self.cum_l = {"train":0,
                        "val":0,
                        "test":0}
        self.cum_R_adv = {"train":0,
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

        if pretrained == 0:
            #self.classifier = handmade.ResNet(handmade.BasicBlock, [2, 2, 2, 2])
            self.classifier = models.resnet18(pretrained=False)
        elif pretrained == 1:
            #self.classifier = handmade.ResNet(handmade.BasicBlock, [2, 2, 2, 2])
            self.classifier = models.resnet18(pretrained=True)

        linear_size = list(self.classifier.children())[-1].in_features
        self.classifier.fc = nn.Linear(linear_size, self.num_classes)

    def change_output(self, num_classes=3):
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

    def compute_adversarial_direction(self, x, logit_x):
        dd = torch.randn(x.size()).to(self.device)
        dd.requires_grad = True

        with _disable_tracking_bn_stats(self):
            for _ in range(self.ip):
                dd.requires_grad = True
                dd = self.xi * l2_normalize(dd)

                logit_hat = self.classifier(x + dd)

                dist = kl_div_with_logit(logit_x, logit_hat)

                dd = torch.autograd.grad(dist, dd)[0]

        r_adv = self.eps * l2_normalize(dd.detach())

        return r_adv

    def generic_step(self, train_batch, batch_idx, step_type):
        torch.set_grad_enabled(True)
        x, y = train_batch
        unlabeled_idx = y == -1

        pred_y = self.classifier(x)
        pred_y = self.softmax(pred_y)
        y[unlabeled_idx] = torch.argmax(pred_y, 1)[unlabeled_idx]

        r_vadv = self.compute_adversarial_direction(x, pred_y)

        pred_adv = self.classifier(x + r_vadv)
        pred_adv = self.softmax(pred_adv)
        R_adv = self.kl_div(pred_y, pred_adv)

        self.optimizers().zero_grad()
        l = self.cross_entropy(pred_y, y)
        loss = l + R_adv * self.a
        self.manual_backward(loss)
        self.accuracy[step_type].update(torch.argmax(pred_y, 1).to('cpu'), y.to('cpu'))
        if self.num_classes==2:
            self.auc[step_type].update(self.softmax(pred_y)[:, 1], y)
            if step_type=="test":
            self.confmat["test"].update(self.softmax(pred_y, 1)[:, 1].to("cpu"), y.to("cpu"))
        elif step_type=="test":
            self.confmat["test"].update(torch.argmax(pred_y, 1).to('cpu'), y.to('cpu'))
        
        self.num_steps[step_type]+=1
        self.cum_loss[step_type]+=loss
        self.cum_l[step_type]+=l
        self.cum_R_adv[step_type]+=R_adv
        self.optimizers().step()
        return {'loss': loss, 'preds': pred_y, "target": y, "l": l}

    def training_epoch_start(self):
        self.cum_loss['train']=0
        self.cum_l['train']=0
        self.cum_R_adv['train']=0
        self.num_steps['train']=0

    def training_step(self, train_batch, batch_idx):
        out = self.generic_step(train_batch, batch_idx, step_type="train")
        return out

    def training_epoch_end(self, outputs):
        self.cum_loss['train'] /= self.num_steps['train']
        self.cum_l['train'] /= self.num_steps['train']
        self.cum_R_adv['train'] /= self.num_steps['train']
        accuracy = self.accuracy['train'].compute()
        self.log("train_loss", self.cum_loss['train'])
        self.log("train_l", self.cum_l['train'])
        self.log("train_R_adv", self.cum_R_adv['train'])
        self.log('train_acc', accuracy, prog_bar=True)
        if self.num_classes==2:
            auc = self.auc['train'].compute()
            self.log('train_auc', auc, prog_bar=True)

    def validation_epoch_start(self):
        self.cum_loss['val']=0
        self.cum_l['val']=0
        self.cum_R_adv['val']=0
        self.num_steps['val']=0

    def validation_step(self, val_batch, batch_idx):
        out = self.generic_step(val_batch, batch_idx, step_type="val")
        return out

    def validation_epoch_end(self, outputs):
        self.cum_loss['val'] /= self.num_steps['val']
        self.cum_l['val'] /= self.num_steps['val']
        self.cum_R_adv['val'] /= self.num_steps['val']
        accuracy = self.accuracy['val'].compute()
        self.log("val_loss", self.cum_loss['val'], prog_bar=True)
        self.log("val_l", self.cum_l['val'])
        self.log("val_R_adv", self.cum_R_adv['val'])
        self.log('val_acc', accuracy, prog_bar=True)
        if self.num_classes==2:
            auc = self.auc['val'].compute()
            self.log('val_auc', auc)

    def test_epoch_start(self):
        self.cum_loss['test']=0
        self.cum_l['test']=0
        self.cum_R_adv['test']=0
        self.num_steps['test']=0

    def test_step(self, test_batch, batch_idx):
        out = self.generic_step(test_batch, batch_idx, step_type="test")
        self.log("test_loss", out['loss'])
        return out

    def test_epoch_end(self, outputs):
        self.cum_loss['test'] /= self.num_steps['test']
        self.cum_l['test'] /= self.num_steps['test']
        self.cum_R_adv['test'] /= self.num_steps['test']
        accuracy = self.accuracy['test'].compute()
        confmat = self.confmat["test"].compute()
        self.log("test_loss", self.cum_loss['test'])
        self.log("test_l", self.cum_l['test'])
        self.log("test_R_adv", self.cum_R_adv['test'])
        self.log('test_acc', accuracy)
        self.log("test_confmat", confmat)
        if self.num_classes==2:
            auc = self.auc['test'].compute()
            self.log('test_auc', auc)
