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


def training(run_name, ssl=False):
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
    warnings.filterwarnings("ignore")

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
                model = VATModel(pretrained=False)
            else:
                model = BaselineModel(pretrained=False, model_type='resnet18', im_size=imsize)

            #summary(model, (3, 128, 128))
            early_stopping = pl.callbacks.early_stopping.EarlyStopping(monitor='train_loss', patience=80, mode='min', min_delta=0.0001, check_on_train_epoch_end=True)
            trainer = pl.Trainer(accelerator="gpu", gpus=1, precision=16, max_epochs=1000, callbacks=[early_stopping])

            trainer.fit(model=model, train_dataloaders=DataLoader(train_data, batch_size=batch_size), val_dataloaders=DataLoader(test_data, batch_size=batch_size))
            trainer.save_checkpoint("./models/run{0}".format(datetime.date.today()))


if __name__ == "__main__":
    training(run_name="Resnet18 no pretraining", ssl=True)
