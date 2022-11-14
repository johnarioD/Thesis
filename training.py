import mlflow
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from models import BaselineModel, VATModel
import preprocessing as data
import mlflow.pytorch as tracker
from sklearn.model_selection import KFold
import numpy as np
from dataset_model import bccDataset
import warnings
from torchsummary import summary
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

PRTRN_NONE = 0
PRTRN_IMNT = 1
PRTRN_LESN = 2


def pretraining():
    data.preprocess("/ISIC2020", False)
    data.preprocess("/ISIC2020", True)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    imsize = 512

    # data
    print("Loading Data")
    images = []

    # load labels
    with open("./data/unprocessed/ISIC_tags.csv", 'r') as metadata:
        df = pd.read_csv(metadata)
        Y = dict(zip(df.id, df.label))

    # load images
    indices = []
    for file in df.id:
        images.append(
            cv2.resize(plt.imread("./data/preprocessed_hairy" + "/ISIC2020/" + file + ".jpg"), [imsize, imsize]))
        indices.append(file)
    X = np.array(images) / 255
    Y = np.array([Y[i] for i in indices])

    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    # warnings
    # warnings.filterwarnings("ignore")

    # model
    print("Spliting K-folds")
    batch_size, k = 32, 0
    cross_val_acc = {'train': 0, 'val': 0, 'test': 0}

    # run initialization
    tracker.autolog(silent=True)

    for train_idx, test_idx in kf.split(X):
        k += 1
        with mlflow.start_run():
            train_data = bccDataset(X=X[train_idx], Y=Y[train_idx])
            test_data = bccDataset(X=X[test_idx], Y=Y[test_idx])

            model = BaselineModel(pretrained='False', num_classes=2)

            # summary(model, (3, 128, 128))
            early_stopping = pl.callbacks.early_stopping.EarlyStopping(monitor='train_loss', patience=80, mode='min',
                                                                       min_delta=0.0001, check_on_train_epoch_end=True)
            trainer = pl.Trainer(accelerator="gpu", gpus=1, precision=16, max_epochs=1000, callbacks=[early_stopping])

            trainer.fit(model=model, train_dataloaders=DataLoader(train_data, batch_size=batch_size),
                        val_dataloaders=DataLoader(test_data, batch_size=batch_size))
            trainer.save_checkpoint("./models/baseline_ISIC_" + str(k) + ".chkpt")


def training(run_name, pretrain=0, ssl=False):
    experiment = mlflow.get_experiment("1")

    # reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    imsize = 512
    n_splits = 10
    batch_size = 32
    split_size = 1/n_splits

    # data
    print("Loading Data")
    X, Y = data.load_train_full(version="hairy", ssl=ssl, image_size=imsize)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)

    # sampling
    class_sample_count = np.unique(Y_test, return_counts=True)[1]
    class_sample_count[0] = sum(class_sample_count[1:])
    weight = 1. / class_sample_count
    sample_weights = torch.from_numpy(weight[Y_test])
    test_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    test_dataloader = DataLoader(bccDataset(X_test, Y_test), batch_size=batch_size, sampler=test_sampler)
    # warnings
    # warnings.filterwarnings("ignore")

    # model
    cross_val_acc = {'train': 0, 'val': 0, 'test': 0}

    # run initialization
    tracker.autolog(silent=True)

    # k-folds
    print("Spliting K-folds")
    for k in range(n_splits):
        k += 1
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=split_size/(1-split_size))

        class_sample_count = np.unique(Y_train, return_counts=True)[1]
        class_sample_count[0] = sum(class_sample_count[1:])
        weight = 1. / class_sample_count
        sample_weights = torch.from_numpy(weight[Y_train])
        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        class_sample_count = np.unique(Y_val, return_counts=True)[1]
        class_sample_count[0] = sum(class_sample_count[1:])
        weight = 1. / class_sample_count
        sample_weights = torch.from_numpy(weight[Y_val])
        val_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_dataloader = DataLoader(bccDataset(X=X_train, Y=Y_train), batch_size=batch_size,sampler=train_sampler)
        val_dataloader = DataLoader(bccDataset(X=X_val, Y=Y_val), batch_size=batch_size, sampler=val_sampler)
        with mlflow.start_run(run_name=run_name + "_" + str(k), experiment_id=experiment.experiment_id):
            if ssl:
                model = VATModel(pretrained='False')
            else:
                if pretrain != PRTRN_LESN:
                    model = BaselineModel(num_classes=2, pretrained=pretrain)
                else:
                    model = BaselineModel.load_from_checkpoint(checkpoint_path="./models/baseline_ISIC_1.chkpt",
                                                               num_classes=2)
                    model.change_output(num_classes=2)

            # summary(model, (3, 128, 128))
            early_stopping = EarlyStopping(monitor='train_loss', patience=200, mode='min', min_delta=0.00, check_on_train_epoch_end=True)

            trainer = pl.Trainer(accelerator="gpu", gpus=1, precision=16, max_epochs=1000, callbacks=[early_stopping])

            trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
            trainer.test(model, dataloaders=test_dataloader)
            cross_val_acc['train'] += model.accuracy['train']
            cross_val_acc['val'] += model.accuracy['val']
            cross_val_acc['test'] += model.accuracy['test']

    cross_val_acc['train'] /= n_splits
    cross_val_acc['val'] /= n_splits
    cross_val_acc['test'] /= n_splits
    print("Cross-Validation Results:\n----------------------------------------------")
    print(f"Train Accuracy: {100 * cross_val_acc['train']}.2f%\n")
    print(f"Validation Accuracy: {100 * cross_val_acc['val']}.2f%\n")
    print(f"Test Accuracy: {100 * cross_val_acc['test']}.2f%\n")


if __name__ == "__main__":
    training(run_name="Resnet18 no pretraining", pretrain=0, ssl=False)
    training(run_name="Resnet18 imnet pretraining", pretrain=1, ssl=False)
    training(run_name="Resnet18 lesion pretraining", pretrain=2, ssl=False)
    # pretraining()
