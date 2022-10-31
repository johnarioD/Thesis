import torch
import torchmetrics
import resnet18_handmade as handmade
from torch import nn
from torchvision import models
import pytorch_lightning as pl
import contextlib
from torch.nn import Softmax
import torch.nn.functional as F
import data_augmentation as augment


class BaselineModel(pl.LightningModule):
    def __init__(self, num_classes, pretrained=0):
        super().__init__()
        self.num_classes = num_classes

        self.cross_entropy = nn.CrossEntropyLoss()
        self.accuracy = {"train": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted'),
                         "test": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted'),
                         "val": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted')}

        if pretrained == 0:
            self.classifier = models.resnet18(pretrained=False)
        elif pretrained == 1:
            self.classifier = models.resnet18(pretrained=True)

        linear_size = list(self.classifier.children())[-1].in_features
        self.classifier.fc = nn.Linear(linear_size, self.num_classes)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def change_output(self, num_classes=3):
        self.num_classes = num_classes

        self.accuracy = {"train": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted'),
                         "test": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted'),
                         "val": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted')}

        linear_size = list(self.classifier.children())[-1].in_features
        self.classifier.fc = nn.Linear(linear_size, self.num_classes)

    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        return self.optimizer

    def generic_step(self, train_batch, batch_idx, step_type):
        x, y = train_batch
        x, y = augment.augment_data(x=x, y=y)
        pred_y = self(x)
        loss = self.cross_entropy(pred_y, y)
        self.accuracy[step_type].update(torch.argmax(pred_y, 1).to('cpu'), y.to('cpu'))
        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self.generic_step(train_batch, batch_idx, step_type="train")
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        accuracy = self.accuracy['train'].compute()
        self.log('train_acc', accuracy, prog_bar=True)

    def validation_step(self, val_batch, batch_idx):
        loss = self.generic_step(val_batch, batch_idx, step_type="val")
        self.log("val_loss", loss, prog_bar=True)

    def validation_epoch_end(self, outputs):
        accuracy = self.accuracy['val'].compute()
        self.log('val_acc', accuracy, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        loss = self.generic_step(test_batch, batch_idx, step_type="test")
        self.log("test_loss", loss)

    def test_epoch_end(self):
        accuracy = self.accuracy['test'].compute()
        self.log('test_acc', accuracy)


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
    d_reshaped = d.view(d.size(0) * d.size(1), -1)
    d_norm = F.normalize(d_reshaped, dim=1, p=2).view(d.size())
    return d_norm


class VATModel(pl.LightningModule):
    def __init__(self, num_classes=3, xi=1e-6, eps=1.0, ip=1, a=1, pretrained=False):
        super().__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.a = a

        self.num_classes = num_classes
        self.automatic_optimization = False

        self.softmax = Softmax(dim=1)
        self.criterion = kl_div_with_logit
        self.accuracy = {"train": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted'),
                         "test": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted'),
                         "val": torchmetrics.Accuracy(num_classes=self.num_classes, average='weighted')}

        self.classifier = models.resnet18(pretrained=pretrained)
        linear_size = list(self.classifier.children())[-1].in_features
        self.classifier.fc = nn.Linear(linear_size, self.num_classes)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        self.classifier(x)

    def configure_optimizers(self):
        return self.optimizer

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

        # Create random unit tensor
        # if batch_idx == 0:
        d = torch.rand(x.shape).to(x.device)
        d.requires_grad = True

        pred_y = self.classifier(x)
        # pred_y = self.softmax(pred_y)
        y[unlabeled_idx] = torch.argmax(pred_y, 1)[unlabeled_idx]
        y = F.one_hot(y, num_classes=self.num_classes).type(torch.float16)
        l = self.criterion(pred_y, y)

        r_vadv = self.compute_adversarial_direction(x, pred_y)

        pred_adv = self.classifier(x + r_vadv)
        #pred_adv = self.softmax(pred_adv)
        R_adv = self.criterion(pred_y, pred_adv)

        # loss = R_adv * self.a
        loss = l + R_adv * self.a
        self.optimizer.zero_grad()
        self.manual_backward(loss)
        self.accuracy[step_type].update(torch.argmax(pred_y, 1).to('cpu'), torch.argmax(y, 1).to('cpu'))
        self.optimizer.step()
        return loss, l, R_adv

    def training_step(self, train_batch, batch_idx):
        loss, l, R_adv = self.generic_step(train_batch, batch_idx, step_type="train")
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_l", l)
        self.log("train_R_adv", R_adv)
        return loss

    def training_epoch_end(self, outputs):
        accuracy = self.accuracy['train'].compute()
        self.log('train_acc', accuracy, prog_bar=True)

    def validation_step(self, val_batch, batch_idx):
        loss, l, R_adv = self.generic_step(val_batch, batch_idx, step_type="val")
        self.log("val_loss", loss)
        self.log("val_l", l)
        self.log("val_R_adv", R_adv)

    def validation_epoch_end(self, outputs):
        accuracy = self.accuracy['val'].compute()
        self.log('val_acc', accuracy, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        loss, l, R_adv = self.generic_step(test_batch, batch_idx, step_type="test")
        self.log("test_loss", loss)
        self.log("test_l", l)
        self.log("test_R_adv", R_adv)

    def test_epoch_end(self):
        accuracy = self.accuracy['test'].compute()
        self.log('test_acc', accuracy)
