"""Download and access machine learning datasets under a common tree.

Download and access machine learning datasets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf
import keras

FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'mnist', 'Dataset name to download')
flags.DEFINE_string('framework', 'keras',
                    'The framework that supports the dataset')
flags.DEFINE_string(
    'path', '/tmp/datasets',
    'Base path under which to store the data in persistent storage')
flags.DEFINE_string('cache_path', '/tmp/datasets.cache',
                    'Temporary path into which to download the data')

_BASE_FILENAMES = {
    'keras': {
        'mnist': 'mnist.npz',
        'cifar10': 'cifar-10-python.tar.gz',
    }
}


def _get_paths(framework, dataset):
  """Return the relevant paths for handling the given dataset.

  Args:
    framework: The ML framework that manages this dataset, e.g. 'keras'
    dataset: The name of the dataset, e.g. 'mnist'

  Returns:
    A tuple containing the following paths.

    (persist_path, persist_directory, cache_path)

    persist_directory: The directory in which the dataset is stored
      in persistent storage.
    persist_path: The dataset path in persistent storage.
    cache_path: The local dataset path used for caching.
  """
  try:
    base_file = _BASE_FILENAMES[framework][dataset]
    persist_dir = '{}/{}/{}'.format(FLAGS.path, framework, dataset)
    persist_path = persist_dir + '/' + base_file
    cache_path = '/' + FLAGS.cache_path + '/' + base_file
    return (persist_path, persist_dir, cache_path)
  except MissingKeyException:
    raise ValueError('Unsupported dataset')


def download_keras_cifar10():
  """Download CIFAR10 dataset as packaged by Keras.

  Download the dataset and save it under --path.
  """
  (persist_path, persist_dir, cache_path) = _get_paths('keras', 'cifar10')
  if tf.gfile.Exists(persist_path):
    return
  tf.gfile.MakeDirs(persist_dir)
  keras.datasets.mnist.load_data(cache_path)
  tf.gfile.Copy(cache_path, persist_path)
  tf.gfile.Remove(cache_path)


def download_keras_mnist():
  """Download MNIST dataset as packaged by Keras.

  Download the dataset and save it under --path.
  """
  (persist_path, persist_dir, cache_path) = _get_paths('keras', 'mnist')
  if tf.gfile.Exists(persist_path):
    return
  tf.gfile.MakeDirs(persist_dir)
  keras.datasets.mnist.load_data(cache_path)
  tf.gfile.Copy(cache_path, persist_path)
  tf.gfile.Remove(cache_path)


def load_keras_mnist():
  """Load MNIST (Keras).

  Returns:
    Data in the same format as keras.datasets.mnist.load_data().

  Raises:
    ValueError: If the data is missing
  """
  (persist_path, persist_dir, cache_path) = _get_paths('keras', 'mnist')
  if not tf.gfile.Exists(persist_path):
    err_msg = 'Expected dataset file {} does not exist'.format(persist_path)
    tf.logging.error(err_msg)
    raise ValueError(err_msg)
  if not tf.gfile.Exists(cache_path):
    tf.gfile.Copy(persist_path, cache_path)
  return keras.datasets.mnist.load_data(cache_path)


def delete_keras_mnist():
  """Delete downloaded MNIST (Keras) dataset."""
  (persist_path, persist_dir, cache_path) = _get_paths('keras', 'mnist')
  if tf.gfile.Exists(cache_path):
    tf.gfile.Remove(cache_path)
  if tf.gfile.Exists(persist_path):
    tf.gfile.Remove(persist_path)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  tf.gfile.MakeDirs(FLAGS.cache_path)
  download_keras_mnist()
  ((x_t, y_t), (x_v, y_v)) = load_keras_mnist()
  print(y_t)


if __name__ == '__main__':
  app.run(main)
