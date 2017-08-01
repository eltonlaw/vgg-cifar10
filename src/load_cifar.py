""" Extract and read CIFAR-10 dataset

Dataset downloaded from: https://www.cs.toronto.edu/~kriz/cifar.html

"""
import os
import tarfile
import pickle
import numpy as np
# pylint: disable=import-error
from preprocess import rotate_reshape
from preprocess import subtract_mean_rgb
from preprocess import rescale


def _extract(path, filename):
    """ Extracts tarfile at path `tarpath` to current working dir"""
    fullpath = os.path.join(path, filename)
    if not tarfile.is_tarfile(fullpath):
        raise Exception("'{}' is not a tarfile".format(fullpath))
    with tarfile.open(name=fullpath) as file:
        file.extractall(path=path)


def _unprocessed_batch(dir_path, filename):
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

def _preprocess(images_1d, labels_1d, n_labels=10, dshape=(32, 32, 3),
                reshape=[224, 224, 3]):
    """ Preprocesses CIFAR10 images

    images_1d: np.ndarray
        Unprocessed images
    labels_1d: np.ndarray
        1d vector of labels
    n_labels: int, 10
        Images are split into 10 classes
    dshape: array, [32, 32, 3]
        Images are 32 by 32 RGB
    """
    labels = _one_hot_encode(labels_1d, n_labels)
    # Reshape and rotate 1d vector into image
    images_raw = rotate_reshape(images_1d, dshape)
    # Rescale images to 224,244
    images_rescaled = rescale(images_raw, reshape)
    # Subtract mean RGB value from every pixel
    images = subtract_mean_rgb(images_rescaled)
    return images, labels

# pylint: disable=too-many-instance-attributes
class Dataset:
    """ Wrapper around CIFAR-10 Dataset"""
    def __init__(self, batch_size, data_path="./cifar-10-batches-py"):
        # There are 50 000 training images
        self.n_data = 50000
        self.n_labels = 10
        if self.n_data % batch_size == 0:
            self._batch_size = batch_size
            self._n_batches = int(self.n_data/batch_size)
        else:
            raise Exception("Datapoints not divisible by batch size")
        self._batch_counter = 0
        self.data = {"train": [], "test": []}
        self.data_path = data_path
        self._setup_tr_complete = False
        # One-time thing, extracts tar file
        # _extract(dir_path, "cifar-10-python.tar.gz")

    def setup_train_batches(self):
        """ Create a list of starting and ending indices for each batch """
        if self._setup_tr_complete:
            raise Exception("Already called `setup_train_batches`")
        train_files = ["data_batch_1", "data_batch_2", "data_batch_3",
                       "data_batch_4", "data_batch_5"]
        images_1d, labels_1d = [], []
        # Training dataset split into 5 parts, concatenate all 5 parts together
        for file in train_files:
            images_temp, labels_temp = _unprocessed_batch(self.data_path, file)
            images_1d.append(images_temp)
            labels_1d.append(labels_temp)
        # Hacky fix to flatten the array into a 2d matrix
        images_1d = np.array(images_1d)
        labels_1d = np.array(labels_1d)
        images_1d = np.reshape(images_1d, (50000, 3072))
        labels_1d = np.reshape(labels_1d, (50000,))
        # Run preprocessing on concatenated (complete) training dataset
        images, labels = _preprocess(images_1d, labels_1d)

        # Split preprocessed dataset into batches according to given batch size
        for i in range(self._n_batches):
            image_batch = images[i*self._batch_size:(i+1)*self._batch_size]
            label_batch = labels[i*self._batch_size:(i+1)*self._batch_size]
            self.data["train"].append(image_batch, label_batch)

        self._setup_tr_complete = True

    # pylint: disable=invalid-name
    def load_n_images(self, n=10, shape=(64, 64, 3)):
        """ Loads two random images from train/test set. Used for testing """
        files = ["data_batch_1", "data_batch_2", "data_batch_3",
                 "data_batch_4", "data_batch_5", "test_batch"]
        random_file = files[np.random.choice(len(files))]
        imgs_1d, labels_1d = _unprocessed_batch(self.data_path, random_file)
        rn = np.random.choice(len(labels_1d))
        images, labels = _preprocess(imgs_1d[rn:rn+n], labels_1d[rn:rn+n],
                                     reshape=shape)
        return images, labels

    def load_train_batch(self):
        """ Load next batch"""
        if not self._setup_tr_complete:
            raise Exception("Run `setup_train_batches` first")
        images, labels = self.data["train"][self._batch_counter]
        self._batch_counter += 1
        if self._batch_counter == self._n_batches:
            self._batch_counter = 0
        return images, labels

    def load_test(self):
        """ Returns test set"""
        file_name = "test_batch"
        images_1d, labels_1d = _unprocessed_batch(self.data_path, file_name)
        images, labels = _preprocess(images_1d, labels_1d)
        return images, labels
