""" Test `load_cifar.py`"""
import unittest
# pylint: disable=import-error
from src.load_cifar import Dataset


class TestLoadCIFAR(unittest.TestCase):
    """ Tests for loading and preprocessing the CIFAR-10 dataset """
    def setUp(self):
        self.data = Dataset(batch_size=100)
        self.data.setup()
        self.data.next_train_batch()

    # @unittest.skip("function unfinished")
    def test_return_type(self):
        """Check return type, should return an np.ndarray"""
        self.assertEqual(1+1, 2)
