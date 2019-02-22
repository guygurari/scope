"""Download and access machine learning datasets under a common tree.

Download and access machine learning datasets. Call download_datasets()
to download all the data, then call load_xxx() to get a particular dataset.

How to add a new dataset:

1. Add the download command to download_datasets(), and make sure
   the download goes to ~/.keras/datasets (the default for Keras
   datasets).

2. Add a load_xxx() command to load the data and return it as a
   tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

3. Add a test case to scope.datasets_test.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gzip
import sys
from six.moves import cPickle

from absl import app
from absl import flags

import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'datasets_dir', '/tmp/datasets',
    'Base dir under which to store the data in persistent storage')
# flags.DEFINE_string('cache_path', '/tmp/datasets.cache',
#                     'Temporary path into which to download the data')
flags.DEFINE_boolean('download', False,
                     'Download the datasets and put in persistent storage.')


def datasets_exist():
  """Check if datasets seem to exist."""
  return tf.gfile.Exists(FLAGS.datasets_dir)


def download_datasets():
  """Download all datasets and copy them to persistent storage."""
  tf.logging.info('Downloading datasets')
  keras.datasets.mnist.load_data()
  keras.datasets.cifar10.load_data()
  keras.datasets.cifar100.load_data()
  keras.datasets.fashion_mnist.load_data()
  keras.datasets.boston_housing.load_data()

  tf.gfile.MakeDirs(FLAGS.datasets_dir)
  local_dataset_dir = os.path.join(os.environ['HOME'], '.keras/datasets')

  for root, subdirs, files in os.walk(local_dataset_dir):
    for filename in files:
      relative_dir = os.path.relpath(root, local_dataset_dir)
      # This causes Google Cloud Storage to create a dir called '.'
      if relative_dir == '.':
        target = os.path.join(FLAGS.datasets_dir, filename)
      else:
        target = os.path.join(FLAGS.datasets_dir, relative_dir, filename)
      source = os.path.join(root, filename)
      tf.logging.info('Copying {} -> {}'.format(source, target))
      if not tf.gfile.Exists(target):
        tf.gfile.MakeDirs(os.path.join(FLAGS.datasets_dir, relative_dir))
        tf.gfile.Copy(source, target)


def load_mnist():
  """Loads the MNIST dataset.

  (Adapted from keras.datasets.mnist.load_data())
  # Arguments
      path: path where to cache the dataset locally
          (relative to ~/.keras/datasets).
  # Returns
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  path = os.path.join(FLAGS.datasets_dir, 'mnist.npz')
  with tf.gfile.GFile(path, 'rb') as f:
    data = np.load(f)
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    return (x_train, y_train), (x_test, y_test)


def _load_cifar_batch(fpath, label_key='labels'):
  """Internal utility for parsing CIFAR data.

  (Adapted from keras.cifar.load_batch())
  # Arguments
      fpath: path the file to parse.
      label_key: key for label data in the retrieve
          dictionary.
  # Returns
      A tuple `(data, labels)`.
  """
  with tf.gfile.GFile(fpath, 'rb') as f:
    if sys.version_info < (3,):
      d = cPickle.load(f)
    else:
      d = cPickle.load(f, encoding='bytes')
      # decode utf8
      d_decoded = {}
      for k, v in d.items():
        d_decoded[k.decode('utf8')] = v
      d = d_decoded
  data = d['data']
  labels = d[label_key]

  data = data.reshape(data.shape[0], 3, 32, 32)
  return data, labels


def load_cifar10():
  """Loads CIFAR10 dataset.

  (Adapted from keras.datasets.cifar10.load_data())
  # Returns
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  dirname = 'cifar-10-batches-py'
  path = os.path.join(FLAGS.datasets_dir, dirname)

  num_train_samples = 50000

  x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
  y_train = np.empty((num_train_samples,), dtype='uint8')

  for i in range(1, 6):
    fpath = os.path.join(path, 'data_batch_' + str(i))
    (x_train[(i - 1) * 10000:i * 10000, :, :, :],
     y_train[(i - 1) * 10000:i * 10000]) = _load_cifar_batch(fpath)

  fpath = os.path.join(path, 'test_batch')
  x_test, y_test = _load_cifar_batch(fpath)

  y_train = np.reshape(y_train, (len(y_train), 1))
  y_test = np.reshape(y_test, (len(y_test), 1))

  if K.image_data_format() == 'channels_last':
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

  return (x_train, y_train), (x_test, y_test)


def load_cifar100(label_mode='fine'):
  """Loads CIFAR100 dataset.

  (Adapted from keras.datasets.cifar100.load_data())
    # Arguments
        label_mode: one of "fine", "coarse".
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    # Raises
        ValueError: in case of invalid `label_mode`.
    """
  if label_mode not in ['fine', 'coarse']:
    raise ValueError('`label_mode` must be one of `"fine"`, `"coarse"`.')

  dirname = 'cifar-100-python'
  path = os.path.join(FLAGS.datasets_dir, dirname)

  fpath = os.path.join(path, 'train')
  x_train, y_train = _load_cifar_batch(fpath, label_key=label_mode + '_labels')

  fpath = os.path.join(path, 'test')
  x_test, y_test = _load_cifar_batch(fpath, label_key=label_mode + '_labels')

  y_train = np.reshape(y_train, (len(y_train), 1))
  y_test = np.reshape(y_test, (len(y_test), 1))

  if K.image_data_format() == 'channels_last':
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

  return (x_train, y_train), (x_test, y_test)


def load_fashion_mnist():
  """Loads the Fashion-MNIST dataset.

    (Adapted from keras.datasets.fashion_mnist.load_data())
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
  dirname = 'fashion-mnist'
  files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
  ]

  paths = [os.path.join(FLAGS.datasets_dir, dirname, fname) for fname in files]

  with tf.gfile.GFile(paths[0], 'rb') as f:
    lbpath = gzip.GzipFile(fileobj=f)
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with tf.gfile.GFile(paths[1], 'rb') as f:
    imgpath = gzip.GzipFile(fileobj=f)
    x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

  with tf.gfile.GFile(paths[2], 'rb') as f:
    lbpath = gzip.GzipFile(fileobj=f)
    y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with tf.gfile.GFile(paths[3], 'rb') as f:
    imgpath = gzip.GzipFile(fileobj=f)
    x_test = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

  return (x_train, y_train), (x_test, y_test)


def load_boston_housing(test_split=0.2, seed=113):
  """Loads the Boston Housing dataset.

    (Adapted from keras.datasets.boston_housing.load_data())
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
        test_split: fraction of the data to reserve as test set.
        seed: Random seed for shuffling the data
            before computing the test split.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
  assert 0 <= test_split < 1
  path = os.path.join(FLAGS.datasets_dir, 'boston_housing.npz')
  with tf.gfile.GFile(path, 'rb') as f:
    data = np.load(f)
    x = data['x']
    y = data['y']

  np.random.seed(seed)
  indices = np.arange(len(x))
  np.random.shuffle(indices)
  x = x[indices]
  y = y[indices]

  x_train = np.array(x[:int(len(x) * (1 - test_split))])
  y_train = np.array(y[:int(len(x) * (1 - test_split))])
  x_test = np.array(x[int(len(x) * (1 - test_split)):])
  y_test = np.array(y[int(len(x) * (1 - test_split)):])
  return (x_train, y_train), (x_test, y_test)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if FLAGS.download:
    download_datasets()


if __name__ == '__main__':
  app.run(main)
