import mlflow
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import mlflow.pytorch as tracker
import model as mdl
import preprocessing as data
import mlflow.pytorch as tracker
from mlflow.tracking import MlflowClient
from sklearn.model_selection import KFold
import numpy as np
import datetime
from dataset_model import bccDataset
import warnings


def train_model():
    # reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # data
    X, Y = data.load_train("data/preprocessed_hairy/BCC", "data/unprocessed/BCC FINAL Learning Set.csv")
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    # warnings
    warnings.filterwarnings("ignore")

    # model
    batch_size, k = 32, 0
    cross_val_train_acc, cross_val_val_acc = 0, 0
    for train_idx, test_idx in kf.split(X):
        print("Things are happening ", k)
        k += 1
        train_data = bccDataset(X=X[train_idx], Y=Y[train_idx])
        test_data = bccDataset(X=X[test_idx], Y=Y[test_idx])
        model = mdl.ResNevus(train_data.class_balance, should_transfer=False)
        early_stopping = pl.callbacks.early_stopping.EarlyStopping(monitor='train_loss', patience=40, mode='min', min_delta=0.0001, check_on_train_epoch_end=True)
        trainer = pl.Trainer(accelerator="gpu", gpus=1, precision=16, max_epochs=1000, callbacks=[early_stopping])
        tracker.autolog(silent=True)

        with mlflow.start_run():
            trainer.fit(model, DataLoader(train_data, batch_size=batch_size), DataLoader(test_data, batch_size=batch_size))
            trainer.save_checkpoint("./models/run{0}".format(datetime.date.today()))
            cross_val_train_acc += model.train_acc/5
            cross_val_val_acc += model.val_acc/5
            print("\n\nACC:\nTraining: ", model.train_acc, "\nValidation: ", model.val_acc, "\n\n")

    print(f"Cross-Validation:\nTrain:{cross_val_train_acc}\nVal:{cross_val_val_acc}")


if __name__ == "__main__":
    train_model()
