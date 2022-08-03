from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class bccDataset(Dataset):
    def __init__(self, X):
        self.X = X.transpose(0, 3, 1, 2).astype(np.float32)

    def __init__(self, X, Y):
        self.X = X.transpose(0, 3, 1, 2).astype(np.float32)
        self.Y = Y.astype(np.int64)-1
        #self.Y = OneHotEncoder(dtype=np.float32).fit_transform(Y.reshape(-1, 1)).toarray()
        self.class_balance = [1,1,1]#np.sum(self.Y, axis=0)
        #self.Y = Y.astype(np.int64)
    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]