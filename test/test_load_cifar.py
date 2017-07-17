""" Test `load_cifar.py`"""
import unittest
import numpy as np
# pylint: disable=import-error
from src.load_cifar import Dataset


class TestLoadCIFAR(unittest.TestCase):
    """ Tests for loading and preprocessing the CIFAR-10 dataset """
    def setUp(self):
        self.data = Dataset(100)

    # @unittest.skip("takes a long time")
    def test_load_test_type(self):
        """ `load_test` returns images and labels array"""
        images, labels = self.data.load_test()
        self.assertTrue(isinstance(images, np.ndarray))
        self.assertTrue(isinstance(labels, np.ndarray))

    # @unittest.skip("takes a long time")
    def test_load_test_shape(self):
        """ Check size of test images/labels array"""
        images, labels = self.data.load_test()
        expected_images_shape = (10000, 224, 224, 3)
        expected_labels_shape = (10000, )
        self.assertEqual(np.shape(images), expected_images_shape)
        self.assertEqual(np.shape(labels), expected_labels_shape)

    def test_load_train_batch_shape(self):
        """Check that it returns the correct batch size """
        self.data.setup_train_batches()
        images, labels = self.data.load_train_batch()
        # Batch size of 100
        expected_images_shape = (100, 224, 224, 3)
        # Labels are one hot encoded, 10 categories
        expected_labels_shape = (100, 10)
        self.assertEqual(np.shape(images), expected_images_shape)
        self.assertEqual(np.shape(labels), expected_labels_shape)
