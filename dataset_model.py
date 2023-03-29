from torch.utils.data import Dataset
import numpy as np
import cv2
from matplotlib import pyplot as plt
import albumentations as A


class bccDataset(Dataset):
    def __init__(self, X, Y, image_size=512):
        self.X = X
        self.Y = Y.astype(np.int64)
        self.imsize = image_size
        self.transform = A.Compose([
            A.Flip(p=0.5),
            A.ShiftScaleRotate(rotate_limit=[-179, 180], scale_limit=0, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2)
            #ToTensorV2()
        ])

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, index):
        image = cv2.resize(plt.imread(self.X[index]), [self.imsize, self.imsize])
        image = image.astype(np.float32)/255
        image = self.transform(image=image)['image']
        return image.transpose(2, 0, 1), self.Y[index]
