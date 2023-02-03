import mlflow
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
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

def testing(run_name, model_path, hair=True, num_classes=2):
    experiment = mlflow.get_experiment("1")

    # reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Training Params
    imsize = 512
    batch_size = 32

    # Non-ssl data
    X_test, Y_test = data.load_train(hair=hair, ssl=False, image_size=imsize, merge_classes=(num_classes==2))
    
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
    with mlflow.start_run(run_name=run_name + "_" + str(k), experiment_id=experiment.experiment_id):
        trainer = pl.Trainer(accelerator="gpu", gpus=1, precision=32, max_epochs=1000)
        model = trainer.test(ckpt_path=model_path)

    model. dataloaders=test_dataloader


if __name__ == "__main__":
    testing(run_name="testing", model_path="D:/MyProjects/python/Diplo/mlruns/1/09b89930fabb409a87a0998ade143f10/artifacts/model_summary.txt")