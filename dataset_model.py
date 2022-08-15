from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class bccDataset(Dataset):
    def __init__(self, X):
        self.X = X.transpose(0, 3, 1, 2).astype(np.float32)

    def __init__(self, X, Y):
        self.X = X.transpose(0, 3, 1, 2).astype(np.float32)
        self.Y = Y.astype(np.int64)-np.min(Y)
        self.class_balance = []#np.sum(self.Y, axis=0)
        for class_id in range(np.min(self.Y), np.max(self.Y)+1):
            self.class_balance.append(np.sum(self.Y==class_id))
        for label in np.unique(self.Y):
            self.class_balance.append(np.count_nonzero(self.Y == label))

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]