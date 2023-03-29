import mlflow
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
import pytorch_lightning as pl
from models import BaselineModel, VotingModel
from ssl_model import VATModel2
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

def testing(run_name, model_path, num_classes, hair=True, n_splits = 10):

    # reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Training Params
    imsize = 512
    batch_size = 32
    split_size = 1/n_splits
    if n_splits < 4:
        split_size = 1/4

    # Model metrics
    cross_val_acc = 0
    if num_classes==2:
        cross_val_auc = 0
    
    # Non-ssl data
    X, Y = data.load_train(hair=hair, ssl=False, image_size=imsize, merge_classes=(num_classes==2))
    for k in range(n_splits):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)

        # Sampling
        weight = len(Y_test)/np.unique(Y_test, return_counts=True)[1]
        sample_weights = torch.from_numpy(weight[Y_test])
        test_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        test_dataset = bccDataset(X_test, Y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

        # Warnings
        # warnings.filterwarnings("ignore")

        # Run Initialization
        tracker.autolog(silent=True)

        print("Model Setup")
        trainer = pl.Trainer(accelerator="gpu", gpus=1, precision=16, max_epochs=1000)
        model = VotingModel(num_classes=num_classes)
        #model = VATModel2(pretrained=0, eps=30, a=1, num_classes=num_classes)
        model = model.load_from_checkpoint(model_path, num_classes=3)
        #model.change_dims(num_classes=num_classes)
        trainer.test(model, dataloaders=test_dataloader)

        cross_val_acc += model.accuracy['test'].compute()
        if k==0:
            confmat = model.confmat['test'].compute()
        else:
            confmat += model.confmat['test'].compute()
        if num_classes==2:
                cross_val_auc += model.auc['test'].compute()

    cross_val_acc /= n_splits
    #confmat /= n_splits
    if num_classes==2:
        cross_val_auc /= n_splits

    print("Cross-Validation Results:\n----------------------------------------------")
    print(f"Accuracy: {100 * cross_val_acc}\n")
    if num_classes==2:
        print(f"AUC: {100 * cross_val_auc}\n")
    if num_classes==2:
        print(f"Confmat:\n\n")
        print(f"{confmat[0][0]}\t{confmat[0][1]}")
        print(f"{confmat[1][0]}\t{confmat[1][1]}")
    elif num_classes==3:
        print(f"Confmat:\n\n")
        print(f"{confmat[0][0]}\t{confmat[0][1]}\t{confmat[0][2]}")
        print(f"{confmat[1][0]}\t{confmat[1][1]}\t{confmat[1][2]}")
        print(f"{confmat[2][0]}\t{confmat[2][1]}\t{confmat[2][2]}")


if __name__ == "__main__":
    testing(run_name="testing", model_path="./lightning_logs/version_839/checkpoints/epoch=999-step=14000.ckpt",num_classes=3)