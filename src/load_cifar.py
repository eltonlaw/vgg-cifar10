""" Extract and read CIFAR-10 dataset

Dataset downloaded from: https://www.cs.toronto.edu/~kriz/cifar.html

"""
import os
import tarfile
import pickle
import numpy as np
# pylint: disable=import-error
from src.preprocess import rotate_reshape
from src.preprocess import subtract_mean_rgb
from src.preprocess import rescale


def _extract(path, filename):
    """ Extracts tarfile at path `tarpath` to current working dir"""
    fullpath = os.path.join(path, filename)
    if not tarfile.is_tarfile(fullpath):
        raise Exception("'{}' is not a tarfile".format(fullpath))
    with tarfile.open(name=fullpath) as file:
        file.extractall(path=path)


def _load_unprocessed_batch(dir_path, filename):
    with open(os.path.join(dir_path, filename), 'rb') as file:
        data = pickle.load(file, encoding='bytes')
    images = data[list(data.keys())[2]]
    labels = data[list(data.keys())[1]]
    return [images, labels]


def _load_meta(path="./cifar-10-batches-py/batches.meta"):
    with open(path, 'rb') as file:
        meta = pickle.load(file, encoding='bytes')
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
    for i, y_i in enumerate(vector):
        one_hot_matrix[i, y_i] = 1
    return one_hot_matrix


# pylint: disable=too-many-instance-attributes
class Dataset:
    """ Wrapper around CIFAR-10 Dataset"""
    def __init__(self, batch_size):
        # There are 50 000 training images
        n_data = 50000
        if n_data % batch_size == 0:
            self._batch_size = batch_size
            self._n_batches = int(n_data/batch_size)
        else:
            raise Exception("Datapoints not divisible by batch size")
        self._batch_counter = 0
        # Images are 32 by 32 RGB
        self._dshape = [32, 32, 3]
        # Images are split into 10 classes
        self._n_labels = 10
        self.data = {"train": [], "test": []}

    def setup(self, data_path="/Users/EltonLaw/data"):
        """ Compiles and loads entire dataset into memory then applies preprocessing"""
        foldername = "cifar-10-batches-py"
        path = os.path.join(data_path, foldername)
        train_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4",
                       "data_batch_5"]
        test_file = "test_batch"
        # One-time thing, extracts tar file
        # _extract(dir_path, "cifar-10-python.tar.gz")
        images_1d, labels_1d = [], []
        for file in train_files:
            images_temp, labels_temp = _load_unprocessed_batch(path, file)
            images_1d.append(images_temp)
            labels_1d.append(labels_temp)
        labels = _one_hot_encode(labels_1d, self._n_labels)
        # Reshape and rotate 1d vector into image
        images_raw = rotate_reshape(images_1d, self._dshape)
        # Rescale images to 224,244
        images_rescaled = rescale(images_raw, [224, 224, 3])
        # Subtract mean RGB value from every pixel
        images = subtract_mean_rgb(images_rescaled)
        train = [[images[i:i+self._batch_size], labels[i:i+self._batch_size]]
                 for i in range(self._n_batches)]
        self.data["train"] = train


    def next_train_batch(self):
        """ Load next batch"""
        images, labels = self.data["train"][self._batch_counter]
        self._batch_counter += 1
        return images, labels
