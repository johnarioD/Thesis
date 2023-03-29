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
import random
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

PRTRN_NONE = 0
PRTRN_IMNT = 1
PRTRN_LESN = 2


def training(run_name, pretrain=PRTRN_NONE, num_classes=2, ssl=False, vote=False, n_splits = 10, hair=True):
    experiment = mlflow.get_experiment("1")

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
    cross_val_acc = {'train': 0, 'val': 0}
    if num_classes==2:
        cross_val_auc = {'train': 0, 'val': 0}
    
    X, Y = data.load_train(hair=hair, ssl=False, image_size=imsize, merge_classes=(num_classes==2))

    # k-folds
    for k in range(n_splits):
        # Non-ssl data
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

        print("Loading Data")
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=split_size/(1-split_size))

        train_dataset = bccDataset(X=X_train, Y=Y_train)
        train_size = len(Y_train)
        train_samples_per_class = np.unique(Y_train, return_counts=True)[1]

        val_dataset = bccDataset(X=X_val, Y=Y_val)
        val_size = len(Y_val)
        val_samples_per_class = np.unique(Y_val, return_counts=True)[1]

        if ssl:
            X, Y = data.load_train(hair=hair, ssl=True, image_size=imsize)
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=split_size)

            train_dataset = ConcatDataset([train_dataset, bccDataset(X=X_train, Y=Y_train)])
            train_size += len(Y_train)
            train_samples_per_class = np.insert(train_samples_per_class, 0, len(Y_train))

            val_dataset = ConcatDataset([val_dataset, bccDataset(X=X_val, Y=Y_val)])
            val_size += len(Y_val)
            val_samples_per_class = np.insert(val_samples_per_class, 0, len(Y_val))

        targets = []
        for _, target in train_dataset:
            if ssl:
                targets.append(target+1)
            else:
                targets.append(target)
        targets = np.array(targets)
        weight = train_size/train_samples_per_class
        sample_weights = torch.from_numpy(weight[targets])
        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        targets = []
        for _, target in val_dataset:
            if ssl:
                targets.append(target+1)
            else:
                targets.append(target)
        targets = np.array(targets)
        weight = val_size/val_samples_per_class
        sample_weights = torch.from_numpy(weight[targets])
        val_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

        print("Model Setup")
        with mlflow.start_run(run_name=run_name + "_" + str(k), experiment_id=experiment.experiment_id):
            if ssl:
                model = VATModel2(pretrained=pretrain, eps=30, a=1, num_classes=num_classes)
            elif vote:
                model = VotingModel(num_classes=num_classes)
            else:
                if pretrain != PRTRN_LESN:
                    model = BaselineModel(num_classes=num_classes, pretrained=pretrain).change_dims()
                else:
                    model = BaselineModel.load_from_checkpoint(checkpoint_path="./models/baseline_ISIC_1.chkpt", num_classes=2)
                    model.change_dims(num_classes=3)

            # summary(model, (3, 128, 128))
            early_stopping = EarlyStopping(monitor='val_loss', patience=200, mode='min', min_delta=0.00, check_on_train_epoch_end=True)

            trainer = pl.Trainer(accelerator="gpu", gpus=1, precision=32, max_epochs=1000, callbacks=[early_stopping])

            trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

            cross_val_acc['train'] += model.accuracy['train'].compute()
            cross_val_acc['val'] += model.accuracy['val'].compute()
            if num_classes==2:
                cross_val_auc['train'] += model.auc['train'].compute()
                cross_val_auc['val'] += model.auc['val'].compute()

    cross_val_acc['train'] /= n_splits
    cross_val_acc['val'] /= n_splits
    if num_classes==2:
        cross_val_auc['train'] /= n_splits
        cross_val_auc['val'] /= n_splits

    with open("data/logs/"+run_name+"_log.txt", 'w')as f:
        results = "Cross-Validation Results:\n----------------------------------------------\n"
        results+=f"Train Accuracy: {100 * cross_val_acc['train']}\n"
        results+=f"Validation Accuracy: {100 * cross_val_acc['val']}\n"
        if num_classes==2:
            results+=f"Train Area Under Curve: {cross_val_auc['train']}\n"
            results+=f"Validation Area Under Curve: {cross_val_auc['val']}\n"
        f.write(results)


if __name__ == "__main__":
    #training(run_name="No Pre Final", pretrain=PRTRN_NONE, ssl=False, num_classes=2, n_splits=1)
    #training(run_name="Imnet Pre Final", pretrain=PRTRN_IMNT, ssl=False, num_classes=2, n_splits=1)
    #training(run_name="Lesion Pre Final", pretrain=PRTRN_LESN, ssl=False, num_classes=2, n_splits=1)
    #training(run_name="No Pre Final", pretrain=PRTRN_NONE, ssl=False, num_classes=3, n_splits=1)
    #training(run_name="Imnet Pre Final", pretrain=PRTRN_IMNT, ssl=False, num_classes=3, n_splits=1)
    #training(run_name="Lesion Pre Final", pretrain=PRTRN_LESN, ssl=False, num_classes=3, n_splits=1)
    training(run_name="Ensemble", pretrain=PRTRN_LESN, vote=True, num_classes=3, n_splits=1)
    #training(run_name="VAT", pretrain=PRTRN_IMNT, ssl=True, num_classes=2, n_splits = 1)
    #training(run_name="VAT", pretrain=PRTRN_IMNT, ssl=True, num_classes=3, n_splits = 1)
