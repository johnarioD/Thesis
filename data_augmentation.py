import cv2
import numpy as np
import random
import torch

def augment_data(x, y):
    x_out = []
    y_out = []
    for image, label in x, y:
        for _ in range(10):
            x_out.append(create_augmented_image(image))
            y_out.append(label)
    return torch.tensor(x_out), torch.tensor(y_out)


def create_augmented_image(image):
    if random.randint(0, 1) == 1:
        tr_vector = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        image = translate(image, tr_vector)
    if random.randint(0, 1) == 1:
        angle = random.randrange(1, 360)
        image = rotate(image, angle)
    if random.randint(0, 1) == 1:
        factor = random.uniform(0.9, 1.1)
        image = luminance_shift(image, factor)

    return image


def translate(image, tr_vector):
    new_image = np.zeros_like(image)
    h, w = image.shape[:2]
    x, y = tr_vector
    x, y = np.floor(x*w), np.floor(y*h)
    new_image[max(x, 0):min(w+x, w), max(y, 0):min(h+y, h), :] = image[max(-x, 0):min(w-x, w), max(-y, 0):min(h-y, h), :]
    return new_image


def rotate(image, angle):
    center = tuple(image.shape[:2]/2)
    T = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(image, T, tuple(image.shape[:2]))


def luminance_shift(image, amount):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)*255
    hsv[:, :, 1] = hsv[:, :, 1] * amount
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2] * amount
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)/255
    return hsv