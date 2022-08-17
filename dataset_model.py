from torch.utils.data import Dataset
import numpy as np


class bccDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.transpose(0, 3, 1, 2).astype(np.float32)
        self.Y = Y.astype(np.int64)-np.min(Y)
        self.class_balance = []
        for label in np.unique(self.Y):
            self.class_balance.append(np.count_nonzero(self.Y == label))

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, index):
        return self.Xl[index], self.Y[index]