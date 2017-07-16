import numpy as np
from scipy.misc import imresize


def rotate_reshape(images, n_data):
    def rr(img):
        img = np.reshape(img, n_data, order="F")
        img = np.rot90(img, k=3)
    new_images = list(map(lambda img: rr(img, n_data)))
    for i, img in enumerate(images):
        img = np.reshape(img, n_data, order="F")
        img = np.rot90(img, k=3)
        new_images.append(img)
    return new_images


def rescale(images, new_size):
    return list(map(lambda img: imresize(img, new_size), images))


def subtract_mean_rgb(images):
    return images - np.uint8(np.mean(images))
