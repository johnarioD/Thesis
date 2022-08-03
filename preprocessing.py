import torchvision
import time
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pandas as pd
from pandas import DataFrame
import re

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
    return np.uint8(255 * image)


def pad_image(image):
    w = max(image.shape[0], image.shape[1])
    padded_image = np.zeros(shape=(w, w, 3), dtype=image.dtype)
    padded_image[:image.shape[0], :image.shape[1]] = image
    return padded_image


# From fitushar/Skin-lesion-Segmentation-using-grabcut
def crop_image_test(image):
    Z = np.float32(image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    kmeans_img = res.reshape((image.shape))

    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

    hsv = cv2.cvtColor(kmeans_img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    h1 = clahe.apply(h)
    s1 = clahe.apply(s)
    v1 = clahe.apply(v)

    lab = cv2.merge((h1, s1, v1))

    Enhance_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(Enhance_img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([100, 255, 255])
    mask_g = cv2.inRange(hsv, lower_green, upper_green)

    ret, inv_mask = cv2.threshold(mask_g, 127, 255, cv2.THRESH_BINARY_INV)

    res = cv2.bitwise_and(image, image, mask=mask_g)

    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    if (np.sum(inv_mask[:]) < 80039400):
        newmask = inv_mask
        mask[newmask == 0] = 0
        mask[newmask == 255] = 1
        dim = cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        GrabCut_img = image * mask2[:, :, np.newaxis]
    else:
        s = (image.shape[0] / 10, image.shape[1] / 10)
        rect = np.uint8((s[0], s[1], image.shape[0] - (3 / 10) * s[0], image.shape[1] - s[1]))
        cv2.grabCut(Enhance_img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        GrabCut_img = image * mask2[:, :, np.newaxis]

    imgmask = cv2.medianBlur(GrabCut_img, 5)
    ret, Segmented_mask = cv2.threshold(imgmask, 0, 255, cv2.THRESH_BINARY)

    if (np.sum(inv_mask[:]) < 80039400):
        newmask = inv_mask
        mask[newmask == 0] = 0
        mask[newmask == 255] = 1
        dim2 = cv2.grabCut(lab, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        GrabCut_img2 = image * mask2[:, :, np.newaxis]
    else:
        cv2.grabCut(lab, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        GrabCut_img2 = image * mask2[:, :, np.newaxis]

    imgmask2 = cv2.medianBlur(GrabCut_img2, 5)
    ret, Segmented_mask2 = cv2.threshold(imgmask2, 0, 255, cv2.THRESH_BINARY)
    return Segmented_mask2


def crop_image(image):
    segment = np.zeros(image.shape[:2], np.uint8)
    segment[0:512, 0:512] = 1
    background_mdl = np.zeros((1, 65), np.float64)
    foreground_mdl = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, segment, (0, 0, image.shape[0], image.shape[1]), background_mdl, foreground_mdl, 20,
                cv2.GC_INIT_WITH_RECT)
    mask = np.where((segment == 2) | (segment == 0), 0, 1).astype('uint8')
    new_image = image * mask[:, :, np.newaxis]
    return new_image


def preprocess(folder, no_hair):
    path_to_original = Path('./data/unprocessed' + folder)
    if no_hair:
        path_to_processed = Path('./data/preprocessed' + folder)
    else:
        path_to_processed = Path('./data/preprocessed_hairy' + folder)

    target_size = [512, 512]
    traverser = os.walk(path_to_original)
    files_processed, prev = 0, 0
    t = time.perf_counter()
    for _, _, files in traverser:
        total_files = len(files)
        for file in files:
            files_processed = files_processed + 1
            if os.path.exists(path_to_processed / file):
                continue

            # f, subplt = plt.subplots(5)
            image = np.array(plt.imread(path_to_original / file))
            # subplt[0].imshow(image)

            resize_mult = np.uint8(max(min(image.shape[0:2]) // 512, 1))
            intermediate_size = [image.shape[1] // resize_mult, image.shape[0] // resize_mult]
            image = cv2.resize(image, intermediate_size)
            # subplt[1].imshow(image)

            image = normalize_color(image)
            # subplt[2].imshow(image)

            if no_hair:
                image = remove_hair(image)

            image = pad_image(image)
            # subplt[3].imshow(image)

            image = cv2.resize(image, target_size)
            # subplt[4].imshow(image)
            # plt.show()

            plt.imsave(path_to_processed / file, image)
            new_t = time.perf_counter()
            if new_t - t >= 60:
                print("\rProgress: {:.2f}% [{:.3f} its/min]".format(100 * files_processed / total_files,
                                                                    60 * (files_processed - prev) / (new_t - t)),
                      end='')
                prev, t = files_processed, new_t
    print("\rProgress: 100%")


def load_train(im_folder, lbl_file=None, image_size=512):
    images, labels = [], DataFrame()

    # load labels
    if lbl_file is not None:
        with open(lbl_file, 'r') as metadata:
            df = pd.read_csv(metadata)
            labels = dict(zip(df.id, df.label))

    # load images
    indices = []
    for _, _, files in os.walk(im_folder):
        for file in files:
            i = int(re.sub('\.jpg|\.JPG', '', file))
            images.append(cv2.resize(plt.imread(im_folder+"/"+file), [image_size, image_size]))
            indices.append(i)
    labels = np.array([labels[i] for i in indices])

    images = np.array(images)/255
    return images, labels


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
    _, _ = load_train("data/preprocessed_hairy/BCC","data/unprocessed/BCC FINAL Learning Set.csv")
