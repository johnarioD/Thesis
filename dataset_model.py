from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from matplotlib import pyplot as plt


class bccDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.transpose(0, 3, 1, 2).astype(np.float32)/255
        self.Y = Y.astype(np.int64)
        self.transform = A.Compose([
            A.Flip(p=0.5),
            A.ShiftScaleRotate(rotate_limit=[-179, 180], scale_limit=0, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2)
            #ToTensorV2()
        ])

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, index):
        image = self.transform(image=self.X[index].transpose(1, 2, 0))['image']
        return image.transpose(2,0,1), self.Y[index]
