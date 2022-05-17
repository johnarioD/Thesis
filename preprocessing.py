import torchvision
import os
from pathlib import Path
from torch.utils.data import Dataset
from skimage.filters import median as median_filter
from skimage.transform import resize
from skimage.color import convert_colorspace
import numpy as np
import matplotlib.pyplot as plt


def normalize_color(image):
    mn = np.min(image)
    mx = np.max(image) - mn
    image = (image - mn)/mx
    return image


def initial_load():
    path_to_original = Path('./data/unprocessed'+"/train")
    path_to_processed = Path('./data/preprocessed'+"/train")

    target_size = [1000, 1000, 3]
    traverser = os.walk(path_to_original)
    files_processed = 0
    for _, directories, files in traverser:
        total_files = len(files)
        for file in files:
            if os.path.exists(path_to_processed / file):
                continue
            files_processed = files_processed + 1
            image = np.array(plt.imread(path_to_original / file))
            image = normalize_color(image)
            image = median_filter(image)
            image = resize(image, target_size)
            plt.imsave(path_to_processed / file, image)
            if files_processed % 100 == 0:
                print("Progress: {:.2f}%".format(100*files_processed/total_files))


def load_train_labeled():
    # load labels
    labels = []
    # load images
    images = []
    return images, labels


def load_train_unlabeled():
    # load images
    images = []
    return images


def load_train_full():
    # load labels
    labels = []
    # load images
    images = []
    return images, labels


def load_test():
    # load labels
    labels = []
    # load images
    images = []
    return images, labels


if __name__ == "__main__":
    initial_load()
