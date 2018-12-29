#!/usr/bin/env python3
"""Train various models on various datasets, and measure quantities related

to the gradient and the Hessian.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import uuid
import os
import sys
import random
import logging
import logging.handlers
import datetime
import tempfile
import time

# python2+3 compatibility
try:
  from StringIO import StringIO
except ImportError:
  from io import StringIO

from absl import app
from absl import flags

import numpy as np
import scipy

import tensorflow as tf
from tensorflow.python.client import device_lib

import keras
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
import colored_traceback

import scope.datasets
import scope.measurements as meas
import scope.models as models
import scope.tbutils as tbutils
import scope.tfutils as tfutils
from scope.experiment_defs import *

colored_traceback.add_hook()

RUN_TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
ALL_SAMPLES = -1

FLAGS = flags.FLAGS

flags.DEFINE_string('job-dir', '', 'Ignored')
flags.DEFINE_string('summary_dir', 'logs', 'Base summary and logs directory')
flags.DEFINE_string(UID_TAG, uuid.uuid4().hex, 'Unique run identifier')
flags.DEFINE_string(NAME_TAG, '', 'Experiment name')
flags.DEFINE_integer('run_number', None,
                     'Number of the current run within the experiment. '
                     'An experiment can have multiple runs with the same '
                     'parameters, for example when collecting statistics.')
flags.DEFINE_integer('total_runs', None,
                     'How many runs total are in the current experiment')
flags.DEFINE_float('delay_logging', None,
                   'Delay log messages going to file but this many seconds '
                   'before flushing.')
flags.DEFINE_boolean('log_to_file', True,
                     'Log everything to file, in addition to stdout. '
                     'If false, only log to stdout.')
flags.DEFINE_string('load_weights', None,
                    'Load the model weights from the given path and use it as '
                    'a starting point')
flags.DEFINE_string('dataset', 'mnist',
                    'Dataset: mnist, mnist_eo, cifar10, sine, or gaussians')
flags.DEFINE_float('image_resize', 1, 'Resize images by given factor')
flags.DEFINE_boolean('dropout', False, 'Use dropout')
flags.DEFINE_boolean('batch_norm', False, 'Use batch normalization')
flags.DEFINE_float('lr', 0.1, 'Learning rate')
flags.DEFINE_float('lr_linear_decay_alpha', None,
                   'alpha defining linear learning rate decay. '
                   'If this is specified, learning rate will start at --lr '
                   'value and will decay linearly with every step until '
                   'step T. The final lr value is alpha*(initial-lr).')
flags.DEFINE_float('lr_linear_decay_T', None,
                   'T defining linear learning rate decay. '
                   'If this is specified, learning rate will start at --lr '
                   'value and will decay linearly with every step until '
                   'step T. The final lr value is alpha*(initial-lr).')

flags.DEFINE_float('resample_prob', 1,
                   'Probability each sample being replaced at each step. '
                   'Must be specified with --iid_batches.')
flags.DEFINE_float('resample_prob_decay_T', None,
                   'T defining resample probability decay. '
                   'If this is specified, it will start at --resample_prob '
                   'value and will decay linearly with every step until '
                   'step T. The final value is alpha*initial.')
flags.DEFINE_float('resample_prob_decay_alpha', None,
                   'alpha defining linear resample probability decay. '
                   'If this is specified, it will start at --resample_prob '
                   'value and will decay linearly with every step until '
                   'step T. The final value is alpha*initial. '
                   'If T is specified and alpha is not, alpha is set to '
                   '1/initial so the final prob value is 1.')

flags.DEFINE_boolean('resample_prob_follows_lr', False,
                     'If True, resample probability will decay with the '
                     'learning rate, starting at --resample_prob.')

flags.DEFINE_boolean('adam', False, 'Use Adam optimizer')
flags.DEFINE_float('momentum', None, 'Use momentum optimizer '
                   '(supply momentum coefficient)')
# flags.DEFINE_boolean('projected_gd', False, 'Use Projected Gradient Descent')
# flags.DEFINE_integer('projected_num_evs', 10,
#                      'How many Hessian eigenvalues to measure for PGD')
# flags.DEFINE_integer('projected_batch_size', 2048,
#                      'Batch size when computing Hessian spectrum for PGD '
#                      '(-1 for all)')
flags.DEFINE_boolean('dense', True,
                     'Include a big fully-connected layer in the CNN')
flags.DEFINE_integer(
    'overparam', 0, 'Overparameterize the dense layers by adding '
    'N linear layers')
flags.DEFINE_string(
    'fc', None, 'Use a fully-connected network, with '
    'hidden layer widths given by comma-separated WIDTHS '
    '(e.g. 100,100). Can also say \'none\' to have '
    'no hidden layers.')
flags.DEFINE_boolean('cnn', False, 'Use a convnet architecture')
flags.DEFINE_integer('cnn_last_layer', 256,
                     'Width of the last dense layer in the CNN')
flags.DEFINE_boolean('small_cnn', False, 'Create a small CNN')
flags.DEFINE_boolean('use_bias', True, 'Use bias in dense layers in FC nets')
flags.DEFINE_string(
    'activation', 'relu',
    'The non-linear activation, can be relu, tanh, softplus,'
    'or anything else supported by Keras.')
flags.DEFINE_float('l2', None, 'L2 weight regularization')
flags.DEFINE_integer('samples', -1, 'Numbers of training samples (-1 for all)')
flags.DEFINE_integer('val_samples', -1,
                     'Numbers of validation samples (-1 for all)')
flags.DEFINE_integer('epochs', 1000, 'Number of training epochs')
flags.DEFINE_integer('steps', None,
                     'Number of training steps (overrides --epochs)')
flags.DEFINE_integer('batch_size', 64, 'Batch size (-1 for all)')
flags.DEFINE_boolean('iid_batches', False,
                     'Sample fully IID batches at each time step '
                     '(allowing replacement)')
flags.DEFINE_boolean('summaries', True, 'Save tensorboard-style summaries')
flags.DEFINE_integer(
    'measure_batch_size', 2048,
    'Batch size used when calculating measurements. Does not '
    'affect results, only performance (-1 for all).')
flags.DEFINE_string(
    'loss_and_acc', '1',
    'Measure loss and accuracy with given frequency '
    '(frequency e.g. "1", "2epochs", "10steps"). '
    'This is more accurate than the builtin Keras measurement.')
flags.DEFINE_string(
    'gradients', None, 'Collect gradient statistics at given frequency '
    '(e.g. "1", "2epochs", "10steps")')
flags.DEFINE_boolean('random_overlap', False,
                     'Compute overlaps of the Hessian with random vectors')
flags.DEFINE_string(
    'hessian', None,
    'Compute a partial Hessian spectrum at given frequency '
    ' (e.g. "1", "2epochs", "10steps")')
flags.DEFINE_string(
    'last_layer_hessian', None,
    'Compute a partial Hessian spectrum, for the last layer weights, '
    ' at given frequency (e.g. "1", "2epochs", "10steps")')
flags.DEFINE_integer('hessian_num_evs', 10,
                     'How many Hessian eigenvalues to measure')
flags.DEFINE_integer('hessian_batch_size', 2048,
                     'Batch size when computing Hessian spectrum (-1 for all)')
flags.DEFINE_string(
    'full_hessian', None, 'Measure the full Hessian spectrum at given frequency'
    ' (e.g. "1", "2epochs", "10steps")')
flags.DEFINE_integer(
    'full_hessian_batch_size', 2048,
    'Batch size when computing the full Hessian. '
    'This may affect performance but does not affect '
    'results. (-1 for all)')
flags.DEFINE_boolean(
    'grad_2pt', False, 'Also collect gradient 2-point functions '
    '(much more expensive)')
flags.DEFINE_string(
    'interpolate_loss', None,
    'Measure interpolated loss in various directions at the given frequency '
    ' (e.g. "1", "2epochs", "10steps")')

flags.DEFINE_integer('gaussians_num_classes', 2,
                     'Number of classes in gaussians dataset')
flags.DEFINE_float('gaussians_sigma', 0, 'Stddev of noise in gaussians dataset')
flags.DEFINE_integer('gaussians_dim', 1000,
                     'Input dimension in gaussians dataset')
flags.DEFINE_boolean('measure_gaussians_every_step', False,
                     'If True, measure gaussians every steps, '
                     'otherwise every epoch')

flags.DEFINE_boolean('show_progress_bar', False,
                     'Show progress bar during training')
flags.DEFINE_string('gpu', '0', 'Which GPU to use')
flags.DEFINE_boolean('cpu', False, 'Use CPU instead of GPU')
flags.DEFINE_integer('seed', None, 'Set the random seed')
flags.DEFINE_boolean('nothing', False, 'Do nothing (for sanity testing)')
flags.DEFINE_boolean('save_final_weights_vector', False,
                     'Save the final trained weights vector as a '
                     'flat vector summary')
flags.DEFINE_integer('prefetch_buffer_size', 10,
                     'How many elements to prefetch during training')


class ExtendedFlags:
  """Wraps the existing FLAGS and allows us to add arbitrary flags during

    runtime.
  """
  def __init__(self):
    self.additional_flags = {}

  def __getattr__(self, name):
    """Gets called if the object does not contain the requested attribute.

        We then pass it on to FLAGS.
        """
    return getattr(FLAGS, name)

  def set(self, flag, value):
    """Set a flag value.

        Args:
            attribute: The flag name.
            value: The flag value.
    """
    setattr(self, flag, value)
    self.additional_flags[flag] = value


xFLAGS = ExtendedFlags()


def get_padded_run_number():
  if xFLAGS.total_runs is not None:
    max_len = len(format(xFLAGS.total_runs))
    return format(xFLAGS.run_number, '0' + str(max_len))
  else:
    return format(xFLAGS.run_number)


def run_name():  # pylint: disable=too-many-branches
  """Returns the experiment name."""
  name = xFLAGS.name

  if xFLAGS.run_number is not None:
    name += '-' + get_padded_run_number()

  name += '-' + xFLAGS.dataset
  name += '-' + xFLAGS.activation

  if xFLAGS.fc is None:
    name += '-cnn'
  else:
    name += '-fc' + xFLAGS.fc
  if xFLAGS.small_cnn:
    name += '-smallcnn'
  if not xFLAGS.dense:
    name += '-nodense'
  if xFLAGS.cnn_last_layer != 256:
    name += '-cnnlast{}'.format(xFLAGS.cnn_last_layer)
  if xFLAGS.overparam > 0:
    name += '-overparam{}'.format(xFLAGS.overparam)
  name += '-' + xFLAGS.optimizer_name
  if xFLAGS.batch_norm:
    name += '-batchnorm'
  else:
    name += '-nobatchnorm'
  if xFLAGS.dropout:
    name += '-dropout'
  else:
    name += '-nodropout'
  # else:
  #     name += '-norandomlabels'
  if xFLAGS.l2 is not None:
    nice_l2 = ('%f' % xFLAGS.l2).rstrip('0')
    if nice_l2.endswith('.'):
      nice_l2 += '0'
    name += '-L2reg{}'.format(nice_l2)
  nice_lr = ('%f' % xFLAGS.lr).rstrip('0')
  name += '-lr{}'.format(nice_lr)
  if xFLAGS.lr_linear_decay_alpha is not None:
    name += '-lrDecay{},{}'.format(
        xFLAGS.lr_linear_decay_alpha,
        xFLAGS.lr_linear_decay_T)
  if xFLAGS.iid_batches:
    name += '-iid'
  name += '-batch{}'.format(xFLAGS.batch_size)
  if xFLAGS.resample_prob < 1:
    name += '-resamp{}'.format(xFLAGS.resample_prob)
  if xFLAGS.resample_prob_follows_lr:
    name += '-resampFollosLR'
  name += '-{}-{}'.format(RUN_TIMESTAMP, xFLAGS.uid)
  if name.startswith('-'):
    name = name[1:]
  return name


def resize_images(images, factor):
  """Resize the images by the given factor.

    Args:
        images: List of images.
        factor: float by which to resize.

    Returns:
        Array of resized images.
    """
  return np.array([scipy.misc.imresize(im, factor) for im in images])


def preprocess_images(load_func):
  """Load image data sets using the given function, and resizes the

    images as necessary.

    Args:
        load_func: Function that loads raw image data and labels and returns
          `(x_train, y_train), (x_test, y_test)`.

    Returns:
        Processed `(x_train, y_train), (x_test, y_test)`.
    """
  (x_train, y_train), (x_test, y_test) = load_func()

  if xFLAGS.image_resize != 1:
    x_train = resize_images(x_train, xFLAGS.image_resize)
    x_test = resize_images(x_test, xFLAGS.image_resize)

  return (x_train, y_train), (x_test, y_test)


def is_regression():
  """Returns whether this is a regression (as opposed to classification)."""
  return xFLAGS.dataset == 'sine'


def init_flags():
  """Validate and initialize some command line flags."""
  possible_sets = ('mnist', 'mnist_eo', 'cifar10', 'sine', 'gaussians')

  def fatal(desc):
    """Report a fatal error and quit."""
    tf.logging.error(desc)
    sys.exit(1)

  if xFLAGS.dataset not in possible_sets:
    fatal('Unsupported dataset {}. '
          'Supported datasets: mnist(_eo), '
          'sine, cifar10, gaussians'.format(xFLAGS.dataset))

  if xFLAGS.fc is None and not xFLAGS.cnn:
    fatal('Must specify either --cnn or --fc')

  if is_regression():
    if xFLAGS.fc is None:
      fatal('Must specify --fc when using regression')
    if xFLAGS.samples == ALL_SAMPLES or xFLAGS.val_samples == ALL_SAMPLES:
      fatal('Must specify --samples and --val-samples when using regression')

  if xFLAGS.gradients is None:
    if xFLAGS.hessian is not None:
      fatal('Must specify --gradients with --hessian')

    if xFLAGS.last_layer_hessian is not None:
      fatal('Must specify --gradients with --hessian')

    if xFLAGS.full_hessian is not None:
      fatal('Must specify --gradients with --full-hessian')

  if xFLAGS.adam:
    xFLAGS.set('optimizer_name', 'adam')
  elif xFLAGS.momentum is not None:
    xFLAGS.set('optimizer_name', 'momentum')
  else:
    xFLAGS.set('optimizer_name', 'sgd')

  if xFLAGS.fc is not None:
    if xFLAGS.fc == 'none':
      xFLAGS.set('fc_widths', list())
    else:
      xFLAGS.set('fc_widths', list(map(int, str(xFLAGS.fc).split(','))))

  if xFLAGS.l2 is None:
    xFLAGS.set('l2_regularizer', None)
  else:
    xFLAGS.set('l2_regularizer', keras.regularizers.l2(xFLAGS.l2))

  if xFLAGS.iid_batches and xFLAGS.steps is None:
    fatal('Must specify --steps with --iid_batches')

  if xFLAGS.resample_prob < 1 and not xFLAGS.iid_batches:
    fatal('Must specify --iid_batches with --resample_prob')


def init_randomness():
  """Seed the random number generators."""
  MAXINT32 = 2**31 - 1
  if xFLAGS.seed is None:
    random.seed()
    xFLAGS.set('seed', random.randint(0, MAXINT32))
  random.seed(xFLAGS.seed)
  np.random.seed(random.randint(0, MAXINT32))
  tf.set_random_seed(random.randint(0, MAXINT32))


def get_data():
  """Create the dataset used for training.

    Returns:
        x_train, y_train, x_test, y_test
    """
  if not scope.datasets.datasets_exist():
    raise IOError("Datasets have not been downloaded. "
                  "Run scope/datasets.py --download")
  num_classes = None
  output_dim = None

  def add_channel_dim(shape):
    """Returns a tensor shape with an extra channel dimension."""
    return list(shape) + [1]

  if xFLAGS.dataset == 'mnist':
    output_dim = num_classes = 10

    (x_train, y_train), (x_test, y_test) = preprocess_images(
        scope.datasets.load_mnist)

    x_train = x_train.reshape(add_channel_dim(x_train.shape))
    x_test = x_test.reshape(add_channel_dim(x_test.shape))
  elif xFLAGS.dataset == 'mnist_eo':
    output_dim = num_classes = 2

    (x_train, y_train), (x_test, y_test) = preprocess_images(
        scope.datasets.load_mnist)

    y_train = y_train % 2
    y_test = y_test % 2

    x_train = x_train.reshape(add_channel_dim(x_train.shape))
    x_test = x_test.reshape(add_channel_dim(x_test.shape))
  elif xFLAGS.dataset == 'sine':
    output_dim = 1

    x_train = np.random.rand(xFLAGS.samples).reshape((-1, 1))
    y_train = np.sin(2 * np.pi * x_train)

    x_test = np.random.rand(xFLAGS.val_samples).reshape((-1, 1))
    y_test = np.sin(2 * np.pi * x_test)
  elif xFLAGS.dataset == 'cifar10':
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = preprocess_images(
        scope.datasets.load_cifar10)
  elif xFLAGS.dataset == 'gaussians':
    # Gaussians centered at random unit vectors with stddev sigma
    if xFLAGS.samples != ALL_SAMPLES:
      raise ValueError('Cannot specify --samples with gaussians dataset')

    n_train = 20000
    n_test = 10000
    output_dim = num_classes = int(xFLAGS.gaussians_num_classes)
    d = int(xFLAGS.gaussians_dim)
    centers = np.random.randn(d, num_classes)
    centers = centers / np.linalg.norm(centers, axis=0)

    def make_gaussian_samples(n):
      """Get `n` samples from a multivariate Gaussian."""
      noise = np.random.randn(n, d) * xFLAGS.gaussians_sigma
      x = noise
      y = np.zeros(n)

      samples_per_class = n // num_classes
      for k in range(num_classes):
        off = k * samples_per_class
        rng = range(off, off + samples_per_class)
        x[rng, :] += np.tile(centers[:, k], (samples_per_class, 1))
        y[rng] = k

      return x, y

    x_train, y_train = make_gaussian_samples(n_train)
    x_test, y_test = make_gaussian_samples(n_test)
  else:
    raise ValueError('Unsupported dataset ' + xFLAGS.dataset)

  if xFLAGS.samples != ALL_SAMPLES:
    n = xFLAGS.samples
    x_train = x_train[:n]
    y_train = y_train[:n]

  if xFLAGS.val_samples != ALL_SAMPLES:
    n = xFLAGS.val_samples
    x_test = x_test[:n]
    y_test = y_test[:n]

  num_samples = len(x_train)

  tf.logging.info('x_train shape: {}'.format(x_train.shape))
  tf.logging.info('y_train shape: {}'.format(y_train.shape))
  tf.logging.info('{} train samples'.format(x_train.shape[0]))
  tf.logging.info('{} test samples'.format(x_test.shape[0]))

  if xFLAGS.batch_size == ALL_SAMPLES:
    xFLAGS.set('batch_size', num_samples)
  if xFLAGS.measure_batch_size == ALL_SAMPLES:
    xFLAGS.set('measure_batch_size', num_samples)
  if xFLAGS.hessian_batch_size == ALL_SAMPLES:
    xFLAGS.set('hessian_batch_size', num_samples)
  if xFLAGS.full_hessian_batch_size == ALL_SAMPLES:
    xFLAGS.set('full_hessian_batch_size', num_samples)
  # if xFLAGS.projected_batch_size == ALL_SAMPLES:
  #     xFLAGS.set('projected_batch_size', num_samples)

  def normalize(x, x_for_mean):
    """Put the data between [xmin, xmax] in a data independent way."""
    mean = np.mean(x_for_mean)
    return (x - mean) / 255

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')

  if xFLAGS.dataset != 'gaussians':
    x_train = normalize(x_train, x_train)
    x_test = normalize(x_test, x_train)

  if not is_regression():
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

  if num_classes is not None:
    xFLAGS.set('num_classes', num_classes)

  if output_dim is not None:
    xFLAGS.set('output_dim', output_dim)

  return x_train, y_train, x_test, y_test


def create_model(input_shape, keras_opt):
  """Returns the Keras model for training."""
  if is_regression():
    model = models.regression_fc_model(xFLAGS, xFLAGS.output_dim)
  elif xFLAGS.fc is not None:
    model = models.classification_fc_model(
      xFLAGS, input_shape, xFLAGS.num_classes)
  elif xFLAGS.small_cnn:
    model = models.classification_small_convnet_model(
        xFLAGS, input_shape, xFLAGS.num_classes)
  else:
    model = models.classification_convnet_model(
        xFLAGS, input_shape, xFLAGS.num_classes)

  if is_regression():
    model.compile(
        loss='mean_squared_error', optimizer=keras_opt, metrics=['mae'])
  else:
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras_opt,
        metrics=['accuracy'])

  tf.logging.info('Model summary:')
  buf = StringIO()
  model.summary(print_fn=lambda s: buf.write(s + '\n'))
  tf.logging.info(buf.getvalue())
  tf.logging.info('Total model parameters: {}'.format(
      tfutils.total_num_weights(model)))

  return model


def add_callbacks(
    callbacks, recorder, model, x_train, y_train, x_test, y_test, lr_schedule):
  """Add measurement callbacks."""

  # TODO convert to Dataset
  def get_batch_makers(batch_size):
    """Returns train and test mini-batch makers."""
    train_batches = tfutils.MiniBatchMaker(x_train, y_train, batch_size)
    test_batches = tfutils.MiniBatchMaker(x_test, y_test, batch_size)
    return train_batches, test_batches

  if xFLAGS.loss_and_acc is not None:
    train_batches, test_batches = get_batch_makers(xFLAGS.measure_batch_size)
    freq = meas.Frequency.from_string(xFLAGS.loss_and_acc)
    loss_acc_cb = meas.BasicMetricsMeasurement(
        recorder,
        model,
        freq,
        train_batches,
        test_batches,
        lr_schedule,
        show_progress=not xFLAGS.show_progress_bar)
    callbacks.append(loss_acc_cb)

    weight_norm_cb = meas.WeightNormMeasurement(recorder, model, freq)
    callbacks.append(weight_norm_cb)

  grad_cb = None
  if xFLAGS.gradients is not None:
    train_batches, test_batches = get_batch_makers(xFLAGS.measure_batch_size)
    freq = meas.Frequency.from_string(xFLAGS.gradients)
    grad_cb = meas.GradientMeasurement(recorder, model, freq, train_batches,
                                       test_batches, xFLAGS.random_overlap)
    callbacks.append(grad_cb)

  if xFLAGS.hessian is not None:
    freq = meas.Frequency.from_string(xFLAGS.hessian)
    hess_cb = meas.LanczosHessianMeasurement(
        recorder,
        model,
        freq,
        xFLAGS.hessian_num_evs,
        x_train,
        y_train,
        xFLAGS.hessian_batch_size,
        lr=xFLAGS.lr,
        log_dir=xFLAGS.runlogdir)
    callbacks.append(hess_cb)

  if xFLAGS.last_layer_hessian is not None:
    freq = meas.Frequency.from_string(xFLAGS.last_layer_hessian)
    if xFLAGS.use_bias:
      weights = model.trainable_weights[-2:]
    else:
      weights = model.trainable_weights[-1:]
    num_weights = tfutils.num_weights(weights)
    grad_subvec = lambda g: g[-num_weights:]
    ll_hess_cb = meas.LanczosHessianMeasurement(
        recorder,
        model,
        freq,
        xFLAGS.hessian_num_evs,
        x_train,
        y_train,
        xFLAGS.hessian_batch_size,
        lr=xFLAGS.lr,
        log_dir=xFLAGS.runlogdir,
        weights=weights,
        grad_subvec=grad_subvec,
        name=meas.LAST_LAYER)
    callbacks.append(ll_hess_cb)

  if xFLAGS.full_hessian is not None:
    train_batches, test_batches = get_batch_makers(
        xFLAGS.full_hessian_batch_size)
    freq = meas.Frequency.from_string(xFLAGS.full_hessian)
    full_hess_cb = meas.FullHessianMeasurement(
        recorder,
        model,
        freq,
        train_batches,
        xFLAGS.runlogdir,
        num_eigenvector_correlations=xFLAGS.output_dim)
    callbacks.append(full_hess_cb)

  if xFLAGS.interpolate_loss is not None:
    train_batches, test_batches = get_batch_makers(xFLAGS.measure_batch_size)
    freq = meas.Frequency.from_string(xFLAGS.interpolate_loss)
    loss_interp_cb = meas.LossInterpolationMeasurement(
        recorder,
        model,
        freq,
        train_batches,
        test_batches)
    callbacks.append(loss_interp_cb)

  if xFLAGS.dataset == 'gaussians':
    freq = meas.Frequency(1, xFLAGS.measure_gaussians_every_step)
    gauss_cb = meas.GaussiansMeasurement(
        recorder, model, freq, x_train, y_train)
    callbacks.append(gauss_cb)


def linear_decay(x0, alpha, T, t):
    """Compute the linear decay rate of quantity x at time t.

    x(t) = x0 - (1-alpha) * x0 * t / T   if t <= T
    x(t) = alpha * x0                    if t > T

    Args:
      x0: Initial value
      alpha: Linear decay coefficient (alpha > 0)
      T: Time at which to stop decaying
      t: Current time
    """
    if t <= T:
      return x0 - (1 - alpha) * x0 * t / T
    else:
      return alpha * x0


class LearningRateLinearDecaySchedule(meas.Measurement):
  """Update the learning rate according to a linear decay schedule,
  and record it. Used for example in 1811.03600."""
  def __init__(self, lr_tensor, eta0, alpha=None, T=None):
    """Ctor. The learning rate at step t will be given by:

    eta(t) = eta0 - (1-alpha) * eta0 * t / T   if t <= T
    eta(t) = alpha * eta0                      if t > T

    Args:
      lr_tensor: Tensor holding the learning rate during training
      eta0: Initial learning rate
      alpha: Linear decay coefficient, or None to keep constant lr
      T: Time at which to stop decaying, or None to keep constant lr
    """
    super(LearningRateLinearDecaySchedule, self).__init__(
        meas.Frequency(freq=1, stepwise=True),
        recorder=None)
    self.lr_tensor = lr_tensor
    self.eta0 = eta0
    self.alpha = alpha
    self.T = T

  def lr(self):
    """Returns the current learning rate"""
    if self.T is None:
      return self.eta0
    else:
      return linear_decay(self.eta0, self.alpha, self.T, self.step)

  def feed_dict(self):
    """Returns a feed_dict with the learning rate filled in."""
    return {self.lr_tensor: self.lr()}


def save_model_weights(model):
  """Save the model."""
  save_dir = xFLAGS.runlogdir + '/saved_models'
  tf.gfile.MakeDirs(save_dir)
  model_filename = '{}-model-weights.h5'.format(xFLAGS.uid)
  model_weights_path = os.path.join(save_dir, model_filename)

  # TODO when h5py 2.9.0 is available, we can create the h5py.File
  #      using a tf.gfile.GFile object and skip the temp file.
  tmp_model_weights_path = os.path.join(tempfile.gettempdir(), model_filename)
  # model.save(tmp_model_weights_path)
  model.save_weights(tmp_model_weights_path)
  tf.gfile.Copy(tmp_model_weights_path, model_weights_path)
  tf.gfile.Remove(tmp_model_weights_path)
  tf.logging.info('Saved trained model at {} '.format(model_weights_path))


def load_model_weights(model_weights_path, model):
  """Load the model weights from the given file.

  Args:
    model_weights_path: Path to model weights file.
    model: An initialized Keras Model.
  """
  tf.logging.info('Loading model weights from {}'.format(model_weights_path))
  tmp_model_weights_path = os.path.join(
      tempfile.gettempdir(), 'model-weights.h5')
  tf.gfile.Copy(model_weights_path, tmp_model_weights_path)
  try:
    # return keras.models.load_model(tmp_model_weights_path)
    model.load_weights(tmp_model_weights_path)
  finally:
    tf.gfile.Remove(tmp_model_weights_path)


def get_optimizer(lr):
  if xFLAGS.optimizer_name == 'sgd':
    keras_opt = keras.optimizers.SGD(lr=lr)
    tf_opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
  elif xFLAGS.optimizer_name == 'adam':
    keras_opt = keras.optimizers.adam()
    tf_opt = tf.train.AdamOptimizer(learning_rate=lr)
  elif xFLAGS.optimizer_name == 'momentum':
    keras_opt = keras.optimizers.SGD(lr=lr, momentum=xFLAGS.momentum)
    tf_opt = tf.train.MomentumOptimizer(
        learning_rate=lr, momentum=xFLAGS.momentum)
  # elif xFLAGS.optimizer_name == 'projected-gd':
  #     hessian_spec = tfutils.KerasHessianSpectrum(
  #         model, x_train, y_train, xFLAGS.projected_batch_size)
  #     keras_opt = meas.ProjectedGradientDescent(
  #         lr, model, x_train, y_train,
  #         hessian_spec, xFLAGS.projected_num_evs)
  else:
    raise RuntimeError('Unknown optimizer: {}'.format(xFLAGS.optimizer_name))
  return keras_opt, tf_opt


def get_learning_rate_schedule(lr_tensor):
  if xFLAGS.lr_linear_decay_alpha is None:
    return LearningRateLinearDecaySchedule(
        lr_tensor, xFLAGS.lr, None, None)
  else:
    return LearningRateLinearDecaySchedule(
        lr_tensor,
        xFLAGS.lr,
        xFLAGS.lr_linear_decay_alpha,
        xFLAGS.lr_linear_decay_T)


class DelayedLoggingHandler(logging.handlers.MemoryHandler):
    """Buffers logging messages and flushes them after a certain delay.
    One use case is writing logs to rate-limited storage.
    """
    def __init__(self, delay, capacity, flushLevel=logging.ERROR, target=None):
        """Ctor.

        Args:
          delay: Seconds to buffer before flushing
          capacity: Buffering capacity
          flushLevel: Automatically flush messages at or above this level
          target: Target logging handler to send records to
        """
        super(DelayedLoggingHandler, self).__init__(
            capacity, flushLevel=flushLevel, target=target)
        self.delay = delay
        self.last_flushed = time.time()

    def _delayPassed(self):
        return time.time() > self.last_flushed + self.delay

    def shouldFlush(self, record):
        """Flush if parent says we should flush, or if enough time elapsed
        since last flush."""
        if super(logging.handlers.MemoryHandler, self).shouldFlush(record):
            self.last_flushed = time.time()
            return True
        elif self._delayPassed():
            self.last_flushed = time.time()
            return True
        else:
            return False


def init_logging():
  xFLAGS.set(RUN_NAME_TAG, run_name())
  xFLAGS.set('runlogdir', '{}/{}'.format(xFLAGS.summary_dir, run_name()))
  tf.gfile.MakeDirs(xFLAGS.runlogdir)
  log = logging.getLogger('tensorflow')
  log.propagate = False
  sh = logging.StreamHandler(sys.stdout)
  log.addHandler(sh)

  # Handle writing to cloud storage
  if xFLAGS.log_to_file:
    fh = logging.StreamHandler(
        tf.gfile.GFile('{}/tensorflow.log'.format(xFLAGS.runlogdir), 'w'))
    if xFLAGS.delay_logging is not None:
        fh = DelayedLoggingHandler(
        delay=xFLAGS.delay_logging,
        capacity=10*1024,
        flushLevel=logging.WARNING,
        target=fh)
    log.addHandler(fh)

  tf.logging.info('Started: {}'.format(RUN_TIMESTAMP))
  tf.logging.info('Log dir: {}'.format(xFLAGS.runlogdir))
  tf.logging.info('Run name: {}'.format(run_name()))

  local_device_protos = device_lib.list_local_devices()
  devices = [x.name for x in local_device_protos]
  tf.logging.info('Available devices: {}'.format(devices))


def get_resample_prob(lr_schedule):
  """Returns the resample probability."""
  if xFLAGS.resample_prob_follows_lr:
    def resample_prob_following_lr():
      lr0 = xFLAGS.lr
      current_prob = np.min([xFLAGS.resample_prob * lr_schedule.lr() / lr0, 1])
      return current_prob
    return resample_prob_following_lr
  elif xFLAGS.resample_prob_decay_T is not None:
    if xFLAGS.resample_prob_decay_alpha is None:
      rp_alpha = 1. / xFLAGS.resample_prob
    else:
      rp_alpha = xFLAGS.resample_prob_decay_alpha
    def resample_prob_decaying():
      return linear_decay(
        xFLAGS.resample_prob,
        rp_alpha,
        xFLAGS.resample_prob_decay_T,
        lr_schedule.step)
    return resample_prob_decaying
  else:
    return xFLAGS.resample_prob


def get_tf_dataset(x, y, lr_schedule):
  """Returns a batch-making Dataset object for the given data."""
  if xFLAGS.iid_batches:
    resample_prob = get_resample_prob(lr_schedule)
    ds = tf.data.Dataset.from_generator(
        tfutils.create_iid_batch_generator(
          x, y, xFLAGS.steps, xFLAGS.batch_size, resample_prob),
        (x.dtype, y.dtype))
  else:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(len(x))
    ds = ds.batch(xFLAGS.batch_size)

  ds = ds.prefetch(buffer_size=xFLAGS.prefetch_buffer_size)
  return ds


def tf_train(sess, x_train, y_train, base_model, tf_opt, lr_schedule, callbacks):
  """TensorFlow training loop."""
  train_ds = get_tf_dataset(x_train, y_train, lr_schedule)
  iterator = tf.data.Iterator.from_structure(train_ds.output_types,
                                             train_ds.output_shapes)
  next_batch = iterator.get_next()
  iterator_init_op = iterator.make_initializer(train_ds)

  model = tfutils.clone_keras_model_shared_weights(
    base_model, next_batch[0], next_batch[1])

  train_step = tf_opt.minimize(model.total_loss)

  sess.run(tf.global_variables_initializer(), feed_dict=lr_schedule.feed_dict())
  step = 0
  epoch = 0

  if xFLAGS.load_weights is not None:
    # Load the model weights instead of the whole model, because Keras
    # has a bug saving/loading models that use InputLayer.
    # https://github.com/keras-team/keras/issues/10417
    #
    # Only load the weights after calling the global variables initializer,
    # otherwise the loaded weights get overwritten.
    load_model_weights(xFLAGS.load_weights, model)

  for callback in callbacks:
    callback.set_model(base_model)

  def counting_epochs():
    return xFLAGS.steps is None

  def should_train_next_epoch():
    if counting_epochs():
      return epoch < xFLAGS.epochs
    else:
      return should_train_next_step()

  def should_train_next_step():
    if counting_epochs():
      return True
    else:
      return step < xFLAGS.steps

  while should_train_next_epoch():
    for callback in callbacks:
      callback.on_epoch_begin(epoch)

    sess.run(iterator_init_op)
    try:
      while should_train_next_step():
        for callback in callbacks:
          callback.on_batch_begin(step)

        feed = tfutils.keras_feed_dict(
            model,
            x=None, y=None,
            feed_dict=lr_schedule.feed_dict(),
            learning_phase=tfutils.KERAS_LEARNING_PHASE_TRAIN)
        sess.run([train_step, model.updates], feed_dict=feed)

        for callback in callbacks:
          callback.on_batch_end(step)
        step += 1
    except tf.errors.OutOfRangeError:
      for callback in callbacks:
        callback.on_epoch_end(epoch)
      epoch += 1


def main(argv):
  if len(argv) > 1:
    print('Unrecognized parameters:', argv)
    sys.exit(1)

  if xFLAGS.nothing:
    sys.exit(0)
  init_flags()
  init_randomness()

  if xFLAGS.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(xFLAGS.gpu)

  x_train, y_train, x_test, y_test = get_data()

  lr_tensor = tf.placeholder(tf.float32, shape=[], name='lr')
  keras_opt, tf_opt = get_optimizer(lr_tensor)

  init_logging()

  tbutils.save_run_flags(
    xFLAGS.runlogdir,
    additional_flags=xFLAGS.additional_flags)

  sess = tf.Session()
  K.set_session(sess)

  lr_schedule = get_learning_rate_schedule(lr_tensor)
  callbacks = [lr_schedule]

  model = create_model(x_train.shape[1:], keras_opt)

  if xFLAGS.summaries:
    recorder = meas.MeasurementsRecorder(summary_dir=xFLAGS.runlogdir)
    add_callbacks(
        callbacks, recorder, model,
        x_train, y_train, x_test, y_test,
        lr_schedule)

  tf.logging.info('Training...')

  tf_train(sess, x_train, y_train, model, tf_opt, lr_schedule, callbacks)

  if xFLAGS.save_final_weights_vector:
    weights = model.get_weights()
    flat_weights = np.concatenate([np.reshape(t, [-1]) for t in weights])
    recorder.record_tensor('final_weights', flat_weights, step=-1)

  if xFLAGS.summaries:
    recorder.close()

  tf.logging.info('Done training!')

  save_model_weights(model)

  # Score trained model.
  scores = model.evaluate(x_test, y_test, verbose=xFLAGS.show_progress_bar)
  tf.logging.info('Test loss: {}'.format(scores[0]))
  tf.logging.info('Test accuracy: {}'.format(scores[1]))


if __name__ == '__main__':
  tf.app.run(main)
