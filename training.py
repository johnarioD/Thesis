import mlflow
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import mlflow.pytorch as tracker
from models import BaselineModel, VATModel
import preprocessing as data
import mlflow.pytorch as tracker
from mlflow.tracking import MlflowClient
from sklearn.model_selection import KFold
import numpy as np
import datetime
from dataset_model import bccDataset
import warnings
from torchsummary import summary
import pandas as pd
import os
import cv2
from matplotlib import pyplot as plt
from pandas import DataFrame
import re


def pretraining():
    data.preprocess("/ISIC2020", False)
    data.preprocess("/ISIC2020", True)

    torch.manual_seed(0)
    np.random.seed(0)

    imsize = 128

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
        images.append(cv2.resize(plt.imread("./data/preprocessed_hairy" + "/ISIC2020/" + file+".jpg"), [imsize, imsize]))
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

            model = BaselineModel(pretrained='False',num_classes=2)

            # summary(model, (3, 128, 128))
            early_stopping = pl.callbacks.early_stopping.EarlyStopping(monitor='train_loss', patience=80, mode='min',
                                                                       min_delta=0.0001, check_on_train_epoch_end=True)
            trainer = pl.Trainer(accelerator="gpu", gpus=1, precision=16, max_epochs=1000, callbacks=[early_stopping])

            trainer.fit(model=model, train_dataloaders=DataLoader(train_data, batch_size=batch_size),
                        val_dataloaders=DataLoader(test_data, batch_size=batch_size))
            trainer.save_checkpoint("./models/baseline_ISIC_"+str(k)+".chkpt")


def training(run_name, pretrain="False", ssl=False):
    experiment = mlflow.get_experiment("1")

    # reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    imsize = 128

    # data
    print("Loading Data")
    X, Y = data.load_train_full(version="hairy", ssl=ssl, image_size=imsize)
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
        with mlflow.start_run(run_name=run_name+"_"+str(k), experiment_id=experiment.experiment_id):
            train_data = bccDataset(X=X[train_idx], Y=Y[train_idx])
            test_data = bccDataset(X=X[test_idx], Y=Y[test_idx])

            if ssl:
                model = VATModel(pretrained='False')
            else:
                model = BaselineModel(pretrained=pretrain)

            #summary(model, (3, 128, 128))
            early_stopping = pl.callbacks.early_stopping.EarlyStopping(monitor='train_loss', patience=80, mode='min', min_delta=0.0001, check_on_train_epoch_end=True)
            trainer = pl.Trainer(accelerator="gpu", gpus=1, precision=16, max_epochs=1000, callbacks=[early_stopping])

            trainer.fit(model=model, train_dataloaders=DataLoader(train_data, batch_size=batch_size), val_dataloaders=DataLoader(test_data, batch_size=batch_size))


if __name__ == "__main__":
    training(run_name="Resnet18 lesion pretraining", pretrain="Lesion", ssl=False)
    #pretraining()
