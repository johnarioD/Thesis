import torchvision
import time
import os
from pathlib import Path
from torch.utils.data import Dataset
from skimage.filters import median as median_filter
from matplotlib import pyplot as plt
import numpy as np
import cv2


# From github.com/sunnyshah2894/DigitalHairRemoval
def remove_hair(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1, (17, 17))
    blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)
    ret, thres2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, thres2, 1, cv2.INPAINT_TELEA)


def normalize_color(image):
    mn = np.min(image)
    mx = np.max(image) - mn
    image = (image - mn) / mx
    return np.uint8(255*image)


def crop_image(image, thresh):
    edges = cv2.Canny(image, 1, 100)
    w, h = edges.shape[0], edges.shape[1]
    lim1, lim2 = 0, max(w, h)
    flag1, flag2 = False, False
    if h > w:
        sm = np.sum(edges, axis=0)
        for i in range(h // 2):
            if sm[i] <= thresh and not flag1:
                lim1 = i
            else:
                flag1 = True
            if sm[h-i-1] <= thresh and not flag2:
                lim2 = h-i-1
            else:
                flag2 = True
            if (lim2-lim1 <= w) or (flag1 and flag2):
                break
        image = image[:, lim1:lim2, :]
    else:
        sm = np.sum(edges, axis=1)
        for i in range(w // 2):
            if sm[i] <= thresh and not flag1:
                lim1 = i
            else:
                flag1 = True
            if sm[w - i - 1] <= thresh and not flag2:
                lim2 = w - i - 1
            else:
                flag2 = True
            if (lim2 - lim1 <= h) or (flag1 and flag2):
                break
        image = image[lim1:lim2, :, :]
    return image


def initial_load():
    path_to_original = Path('./data/unprocessed' + "/train")
    path_to_processed = Path('./data/preprocessed' + "/train")

    target_size = [512, 512]  # 512, 512
    traverser = os.walk(path_to_original)
    files_processed, prev = 0, 0
    t = time.perf_counter()
    for _, directories, files in traverser:
        total_files = len(files)
        for file in files:

            files_processed = files_processed + 1
            if os.path.exists(path_to_processed / file):
                continue
            image = np.array(plt.imread(path_to_original / file))

            resize_mult = np.uint8(max(min(image.shape[0:2])//512, 1))
            intermediate_size = [0, 0]
            intermediate_size[0] = image.shape[1]//resize_mult
            intermediate_size[1] = image.shape[0]//resize_mult
            image = cv2.resize(image, intermediate_size)

            image = normalize_color(image)

            image = remove_hair(image)

            image = median_filter(image)

            image = crop_image(image, 500)

            image = cv2.resize(image, target_size)
            plt.imsave(path_to_processed / file, image)
            if files_processed % 100 == 0:
                new_t = time.perf_counter()
                print("Progress: {:.2f}% [{:.3f} its/min]".format(100 * files_processed / total_files, 60*(files_processed-prev)/(new_t - t)))
                prev = files_processed
                t = new_t


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
