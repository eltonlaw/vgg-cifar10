""" Extract and read CIFAR-10 dataset

Dataset downloaded from: https://www.cs.toronto.edu/~kriz/cifar.html

"""
import tarfile
import pickle
import numpy as np
from src.preprocess import rotate_reshape
from src.preprocess import subtract_mean_rgb
from src.preprocess import rescale


def extract(tarpath):
    if not tarfile.is_tarfile(tarpath):
        raise Exception("'{}' is not a tarfile".format(tarpath))
    with tarfile.open(name=tarpath) as f:
        files = f.extractall()
    return files


def _load_unprocessed_batch(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    images = data[list(data.keys())[2]]
    labels = data[list(data.keys())[1]]
    return images, labels


def _load_meta(path="./cifar-10-batches-py/batches.meta"):
    with open(path, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    labels_bytes = meta[list(meta.keys())[1]]
    labels = [l.decode("utf-8") for l in labels_bytes]
    return labels


def _one_hot_encode(vector, n_unique):
    """ One hot encode a 1D vector

    PARAMETERS
    ----------
    vector: 1D vector. shape: (?, )
    n_unique: Total number of unique values in array

    RETURNS
    -------
    Each label is now a vector with 0's everywhere except a 1 at i=label
    Shape:(?, n_label)
    """
    one_hot_matrix = np.zeros((np.shape(vector)[0], n_unique))
    for i, y in enumerate(vector):
        one_hot_matrix[i, y] = 1
    return one_hot_matrix


class Dataset:
    def __init__(self, datapath, batch_size):
        self.datapath = datapath
        # There are 50 000 training images
        n_data = 50000
        if n_data % batch_size == 0:
            self.bs = batch_size
            self.n_batches = int(n_data/batch_size)
        else:
            raise Exception("Datapoints not divisible by batch size")
        self.batch_counter = 0
        # Images are 32 by 32 RGB
        self.dshape = [32, 32, 3]
        # Images are split into 10 classes
        self.n_labels = 10

    def setup(self):
        images_1d, labels_1d = _load_unprocessed_batch(self.datapath)
        labels = _one_hot_encode(labels_1d, self.n_labels)
        # Reshape and rotate 1d vector into image
        images_raw = rotate_reshape(images_1d, self.dshape)
        # Rescale images to 224,244
        images_rescaled = rescale(images_raw, [224, 224, 3])
        # Subtract mean RGB value from every pixel
        images = subtract_mean_rgb(images_rescaled)
        self.batched_data = [[images[i:i+self.bs], labels[i:i+self.bs]]
                             for i in range(self.n_batches)]

    def load_batch(self):
        images, labels = self.batched_data[self.batch_counter]
        self.batch_counter += 1
        return images, labels
