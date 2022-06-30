from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random


class bccDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]