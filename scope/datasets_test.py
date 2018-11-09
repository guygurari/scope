from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import tensorflow as tf
import keras
import scope.datasets

import colored_traceback
colored_traceback.add_hook()

precision = 5


class TestDatasets(unittest.TestCase):
  def test_keras_mnist(self):
    scope.datasets.delete_keras_mnist()
    scope.datasets.download_keras_mnist()
    ((x_train, y_train), (x_test, y_test)) = scope.datasets.load_keras_mnist()
    ((x_train_ex, y_train_ex), (x_test_ex, y_test_ex)) = keras.datasets.mnist.load_data()
    self.assertTrue(np.all(x_train == x_train_ex))
    self.assertTrue(np.all(y_train == y_train_ex))
    self.assertTrue(np.all(x_test == x_test_ex))
    self.assertTrue(np.all(y_test == y_test_ex))

    scope.datasets.delete_keras_mnist()
    with self.assertRaises(ValueError):
      tf.logging.set_verbosity(tf.logging.FATAL)
      scope.datasets.load_keras_mnist()
      tf.logging.set_verbosity(tf.logging.INFO)

  def test_keras_cifar10(self):
    scope.datasets.download_keras_cifar10()


def main(_):
  unittest.main()


if __name__ == '__main__':
  tf.app.run(main)
