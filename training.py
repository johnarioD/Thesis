import mlflow
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import mlflow.pytorch as tracker
import model as mdl
import preprocessing as data
import mlflow.pytorch as tracker
from mlflow.tracking import MlflowClient
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import datetime
from dataset_model import bccDataset


def train_model():
    # data
    X, Y = data.load_train("data/preprocessed_hairy/BCC", "data/unprocessed/BCC FINAL Learning Set.csv")
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    # model
    model = mdl.LitAutoEncoder()
    batch_size, k = 64, 0
    for train_idx, test_idx in kf.split(X):
        print("Things are happening ", k)
        k += 1
        train_data = bccDataset(X=X[train_idx], Y=Y[train_idx])
        test_data = bccDataset(X=X[test_idx], Y=Y[test_idx])
        trainer = pl.Trainer(accelerator="gpu", gpus=1, precision=16, max_epochs=100)
        tracker.autolog()

        with mlflow.start_run() as run:
            trainer.fit(model, DataLoader(train_data), DataLoader(test_data))
            eval_Y = trainer.predict(model, test_Y)
            print("Split ", k, ": ", accuracy_score(model, test_Y, eval_Y))
            trainer.save_checkpoint("./models/run{0}".format(datetime.date.today()))


if __name__ == "__main__":
    train_model()
