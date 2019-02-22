from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import scope.datasets

import colored_traceback
colored_traceback.add_hook()

precision = 5


class TestDatasets(unittest.TestCase):

  def _load_and_compare(self, expected_loader, actual_loader):
    ((x_train_ex, y_train_ex),
     (x_test_ex, y_test_ex)) = expected_loader()
    ((x_train, y_train), (x_test, y_test)) = actual_loader()
    self.assertTrue(np.all(x_train == x_train_ex))
    self.assertTrue(np.all(y_train == y_train_ex))
    self.assertTrue(np.all(x_test == x_test_ex))
    self.assertTrue(np.all(y_test == y_test_ex))

  def test_mnist(self):
    scope.datasets.download_datasets()
    self._load_and_compare(
        keras.datasets.mnist.load_data, scope.datasets.load_mnist)

  def test_cifar10(self):
    scope.datasets.download_datasets()
    self._load_and_compare(
        keras.datasets.cifar10.load_data, scope.datasets.load_cifar10)

  def test_cifar100(self):
    scope.datasets.download_datasets()
    self._load_and_compare(
        keras.datasets.cifar100.load_data, scope.datasets.load_cifar100)

  def test_fashion_mnist(self):
    scope.datasets.download_datasets()
    self._load_and_compare(
        keras.datasets.fashion_mnist.load_data, scope.datasets.load_fashion_mnist)

  def test_boston_housing(self):
    scope.datasets.download_datasets()
    self._load_and_compare(
        keras.datasets.boston_housing.load_data, scope.datasets.load_boston_housing)


def main(_):
  unittest.main()


if __name__ == '__main__':
  tf.app.run(main)
