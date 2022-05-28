import mlflow
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import mlflow.pytorch as tracker
import model as mdl
import preprocessing as data
import mlflow.pytorch as tracker
from mlflow.tracking import MlflowClient
import datetime


def train_model():
    # data
    train_X, train_Y = data.load_train_labeled()
    eval_X = data.load_train_unlabeled()

    train_loader = DataLoader(train_X, batch_size=32)
    val_loader = DataLoader(train_Y, batch_size=32)

    # model
    model = mdl.LitAutoEncoder()

    # training
    trainer = pl.Trainer(accelerator="gpu", gpus=1, precision=16, max_epochs=100)

    tracker.autolog()

    with mlflow.start_run() as run:
        trainer.fit(model, train_loader, val_loader)
        eval_Y = trainer.predict(model, eval_X)
        trainer.fit(model, eval_X, eval_Y)
        trainer.save_checkpoint("./models/run{0}".format(datetime.date.today()))


if __name__ == "__main__":
    train_model()
