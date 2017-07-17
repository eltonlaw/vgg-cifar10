""" Test `preprocess.py`"""
import unittest
import numpy as np
# -pylint: disable=import-error
from src.preprocess import rotate_reshape
from src.preprocess import rescale
from src.preprocess import subtract_mean_rgb

class TestLoadCIFAR(unittest.TestCase):
    """ Tests for preprocessing the CIFAR-10 dataset """
    def setUp(self):
        self.images_1d = np.ones((10, 3072))
        self.images_3d = np.ones((10, 32, 32, 3))

    def test_rotate_reshape(self):
        """Check output shape, 1D -> 3D array"""
        output_shape = (32, 32, 3)
        output = rotate_reshape(self.images_1d, output_shape)
        # Ignore first index because it represent number of images
        self.assertEqual(np.shape(output)[1:], output_shape)

    def test_rescale(self):
        """ Check that rescale actually rescales to correct dimensions"""
        output_shape = (224, 224, 3)
        output = rescale(self.images_3d, output_shape)
        # Ignore first index because it represent number of images
        self.assertEqual(np.shape(output)[1:], output_shape)

    def test_subtract_mean_rgb(self):
        """ Check that mean of array is subtracted from all values"""
        arr_in = np.array([[1., 2.], [3., 4.]])
        fn_out = subtract_mean_rgb(arr_in)
        # Output should be arr_in - 2.5, with everything rounded to nearest int
        expected_out = np.array([[-1., 0.], [1., 2.]])
        self.assertTrue(np.array_equal(expected_out, fn_out))
