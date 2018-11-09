#!/usr/bin/env python3

"""Train various models on various datasets, and measure quantities related
to the gradient and the Hessian.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import logging
import datetime

from absl import app
from absl import flags

import numpy as np
import scipy

import tensorflow as tf
from tensorflow.python.client import device_lib

import keras
import keras.datasets
from keras.preprocessing.image import ImageDataGenerator
import colored_traceback

import scope.tfutils as tfutils
import scope.tbutils as tbutils
import scope.measurements as meas
import scope.models as models

colored_traceback.add_hook()

RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
ALL_SAMPLES = -1

FLAGS = flags.FLAGS

# def my_DEFINE_string(name, default, help):  # pylint: disable=invalid-name

flags.DEFINE_string('job-dir', '', 'Ignored')
flags.DEFINE_string('logdir', 'logs', 'Base logs directory')
flags.DEFINE_string('name', '', 'Experiment name')
flags.DEFINE_string('dataset', 'mnist',
                    'Dataset: mnist, mnist_eo, cifar10, sine, or gaussians')
flags.DEFINE_float('image_resize', 1, 'Resize images by given factor')
flags.DEFINE_boolean('dropout', False, 'Use dropout')
flags.DEFINE_boolean('batch_norm', False, 'Use batch normalization')
flags.DEFINE_float('lr', 0.1, 'Learning rate')
flags.DEFINE_float('lr_decay', 1,
                   'Multiply the learning rate by this after every epoch ')

flags.DEFINE_boolean('adam', False, 'Use Adam optimizer')
flags.DEFINE_boolean('rmsprop', False, 'Use RMSProp optimizer')
flags.DEFINE_boolean('projected_gd', False, 'Use Projected Gradient Descent')
flags.DEFINE_integer('projected_num_evs', 10,
                     'How many Hessian eigenvalues to measure for PGD')
flags.DEFINE_integer('projected_batch_size', 2048,
                     'Batch size when computing Hessian spectrum for PGD '
                     '(-1 for all)')
flags.DEFINE_boolean('dense', True,
                     'Include a big fully-connected layer in the CNN')
flags.DEFINE_integer('overparam', 0,
                     'Overparameterize the dense layers by adding '
                     'N linear layers')
flags.DEFINE_string('fc', None,
                    'Use a fully-connected network instead of a CNN, with '
                    'hidden layer widths given by comma-separated WIDTHS '
                    '(e.g. 100,100). Can also say \'none\' to have '
                    'no hidden layers.')
flags.DEFINE_boolean('use_bias', True,
                     'Use bias in dense layers in FC nets')
flags.DEFINE_integer('cnn_last_layer', 256,
                     'Width of the last dense layer in the CNN')
flags.DEFINE_boolean('small_cnn', False, 'Create a small CNN')
flags.DEFINE_string('activation', 'relu',
                    'The non-linear activation, can be relu, tanh, softplus,'
                    'or anything else supported by Keras.')
flags.DEFINE_float('l2', None, 'L2 weight regularization')
flags.DEFINE_integer('samples', -1,
                     'Numbers of training samples (-1 for all)')
flags.DEFINE_integer('val_samples', -1,
                     'Numbers of validation samples (-1 for all)')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs')
flags.DEFINE_integer('batch_size', 64,
                     'Batch size (-1 for all)')
flags.DEFINE_boolean('summaries', True, 'Save tensorboard summaries')
flags.DEFINE_boolean('measure_every_step', False,
                     'For every epoch where we measure, measure every '
                     'step (every mini-batch) in the epoch instead of '
                     'only once at the beginning.')
flags.DEFINE_integer('measure_batch_size', 2048,
                     'Batch size used when calculating measurements. Does not '
                     'affect results, only performance (-1 for all).')
flags.DEFINE_integer('loss_and_acc', 1,
                     'Measure loss and accuracy with given frequency. This is '
                     'more accurate than the builtin Keras measurement.')
flags.DEFINE_integer('gradients', None,
                     'Collect gradient statistics every given '
                     'number of epochs')
flags.DEFINE_boolean('random_overlap', False,
                     'Compute overlaps of the Hessian with random vectors')
flags.DEFINE_integer('hessian', None,
                     'Compute a partial Hessian spectrum every FREQ epochs')
flags.DEFINE_integer('hessian_num_evs', 10,
                     'How many Hessian eigenvalues to measure')
flags.DEFINE_integer('hessian_batch_size', 2048,
                     'Batch size when computing Hessian spectrum (-1 for all)')
flags.DEFINE_integer('full_hessian', None,
                     'Measure the full Hessian spectrum '
                     'every this many epochs')
flags.DEFINE_integer('full_hessian_batch_size', 2048,
                     'Batch size when computing the full Hessian. '
                     'This may affect performance but does not affect '
                     'results. (-1 for all)')
flags.DEFINE_boolean('grad_2pt', False,
                     'Also collect gradient 2-point functions '
                     '(much more expensive)')

flags.DEFINE_integer('gaussians_num_classes', 2,
                     'Number of classes in gaussians dataset')
flags.DEFINE_float('gaussians_sigma', 0,
                   'Stddev of noise in gaussians dataset')
flags.DEFINE_integer('gaussians_dim', 1000,
                     'Input dimension in gaussians dataset')

flags.DEFINE_boolean('show_progress_bar', False,
                     'Show progress bar during training')
flags.DEFINE_integer('gpu', 0, 'Which GPU to use')
flags.DEFINE_integer('seed', None, 'Set the random seed')
flags.DEFINE_boolean('nothing', False, 'Do nothing (for sanity testing)')


class ExtendedFlags:
    """Wraps the existing FLAGS and allows us to add arbitrary flags during
    runtime."""
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


xFLAGS = ExtendedFlags()


def run_name():  # pylint: disable=too-many-branches
    """Returns the experiment name."""
    name = xFLAGS.name

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
    if xFLAGS.measure_every_step:
        name += '-everystep'
    # else:
    #     name += '-norandomlabels'
    if xFLAGS.l2 is not None:
        nice_l2 = ('%f' % xFLAGS.l2).rstrip('0')
        if nice_l2.endswith('.'):
            nice_l2 += '0'
        name += '-L2reg{}'.format(nice_l2)
    nice_lr = ('%f' % xFLAGS.lr).rstrip('0')
    name += '-lr{}'.format(nice_lr)
    name += '-batch{}'.format(xFLAGS.batch_size)
    name += '-{}'.format(RUN_TIMESTAMP)
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
    return np.array(
        [scipy.misc.imresize(im, factor) for im in images])


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

    if xFLAGS.dataset not in possible_sets:
        tf.logging.info('Unsupported dataset {}. '
                        'Supported datasets: mnist(_eo), '
                        'sine, cifar10, gaussians'.format(
                            xFLAGS.dataset))
        sys.exit(1)

    if is_regression():
        if xFLAGS.fc is None:
            tf.logging.error('Must specify --fc when using regression')
            sys.exit(1)
        if xFLAGS.samples == ALL_SAMPLES or xFLAGS.val_samples == ALL_SAMPLES:
            tf.logging.error('Must specify --samples and --val-samples '
                             'when using regression')
            sys.exit(1)

    if xFLAGS.gradients is None:
        if xFLAGS.hessian is not None:
            tf.logging.error('Must specify --gradients with --hessian')
            sys.exit(1)

        if xFLAGS.full_hessian is not None:
            tf.logging.error('Must specify --gradients with --full-hessian')
            sys.exit(1)

    if xFLAGS.rmsprop:
        xFLAGS.set('optimizer_name', 'rmsprop')
    elif xFLAGS.adam:
        xFLAGS.set('optimizer_name', 'adam')
    elif xFLAGS.projected_gd:
        xFLAGS.set('optimizer_name', 'projected-gd')
    else:
        xFLAGS.set('optimizer_name', 'sgd')

    if xFLAGS.fc is not None:
        if xFLAGS.fc == 'none':
            xFLAGS.set('fc_widths', list())
        else:
            xFLAGS.set('fc_widths',
                       list(map(int, str(xFLAGS.fc).split(','))))

    if xFLAGS.l2 is None:
        xFLAGS.set('l2_regularizer', None)
    else:
        xFLAGS.set('l2_regularizer', keras.regularizers.l2(xFLAGS.l2))


def init_randomness():
    """Seed the random number generators."""
    MAXINT32 = 2**31 - 1
    if xFLAGS.seed is None:
        random.seed()
        seed = random.randint(0, MAXINT32)
    else:
        seed = xFLAGS.seed
    random.seed(seed)
    np.random.seed(random.randint(0, MAXINT32))
    tf.set_random_seed(random.randint(0, MAXINT32))


def get_dataset():
    """Create the dataset used for training.

    Returns:
        x_train, y_train, x_test, y_test
    """
    num_classes = None
    output_dim = None

    def add_channel_dim(shape):
        """Returns a tensor shape with an extra channel dimension."""
        return list(shape) + [1]

    if xFLAGS.dataset == 'mnist':
        output_dim = num_classes = 10

        (x_train, y_train), (x_test, y_test) = preprocess_images(
            keras.datasets.mnist.load_data)

        x_train = x_train.reshape(add_channel_dim(x_train.shape))
        x_test = x_test.reshape(add_channel_dim(x_test.shape))
    elif xFLAGS.dataset == 'mnist_eo':
        output_dim = num_classes = 2

        (x_train, y_train), (x_test, y_test) = preprocess_images(
            keras.datasets.mnist.load_data)

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
            keras.datasets.cifar10.load_data)
    elif xFLAGS.dataset == 'gaussians':
        # Gaussians centered at random unit vectors with stddev sigma
        if xFLAGS.samples != ALL_SAMPLES:
            raise ValueError('Cannot specify --samples with gaussians dataset')

        n_train = 20000
        n_test = 10000
        num_classes = int(xFLAGS.gaussians_num_classes)
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
    if xFLAGS.projected_batch_size == ALL_SAMPLES:
        xFLAGS.set('projected_batch_size', num_samples)

    def normalize(x, x_for_mean):
        '''Put the data between [xmin, xmax] in a data independent way.'''
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


def get_model(input_shape):
    """Returns the Keras model for training."""
    if is_regression():
        return models.regression_fc_model(xFLAGS, xFLAGS.output_dim)
    elif xFLAGS.fc is not None:
        return models.classification_fc_model(
            xFLAGS, input_shape, xFLAGS.num_classes)
    elif xFLAGS.small_cnn:
        return models.classification_small_convnet_model(
            xFLAGS, input_shape, xFLAGS.num_classes)
    return models.classification_convnet_model(
        xFLAGS, input_shape, xFLAGS.num_classes)


def add_callbacks(
        callbacks, writer, model,
        x_train, y_train, x_test, y_test):
    """Add measurement callbacks."""
    # TODO convert to Dataset
    def get_batch_makers(batch_size):
        """Returns train and test mini-batch makers."""
        train_batches = tfutils.MiniBatchMaker(
            x_train, y_train, batch_size)
        test_batches = tfutils.MiniBatchMaker(
            x_test, y_test, batch_size)
        return train_batches, test_batches

    if xFLAGS.loss_and_acc is not None:
        train_batches, test_batches = get_batch_makers(
            xFLAGS.measure_batch_size)
        freq = meas.MeasurementFrequency(
            xFLAGS.loss_and_acc, xFLAGS.measure_every_step)
        loss_acc_cb = meas.BasicMetricsMeasurement(
            writer, model, freq,
            train_batches, test_batches,
            show_progress=not xFLAGS.show_progress_bar)
        callbacks.append(loss_acc_cb)

        weight_norm_cb = meas.WeightNormMeasurement(
            writer, model, freq)
        callbacks.append(weight_norm_cb)

    grad_cb = None
    if xFLAGS.gradients is not None:
        train_batches, test_batches = get_batch_makers(
            xFLAGS.measure_batch_size)
        freq = meas.MeasurementFrequency(
            xFLAGS.gradients, xFLAGS.measure_every_step)
        grad_cb = meas.GradientMeasurement(
            writer, model, freq,
            train_batches, test_batches, xFLAGS.random_overlap)
        callbacks.append(grad_cb)

    if xFLAGS.hessian is not None:
        freq = meas.MeasurementFrequency(
            xFLAGS.hessian, xFLAGS.measure_every_step)
        hess_cb = meas.LanczosHessianMeasurement(
            writer, model, freq,
            xFLAGS.hessian_num_evs,
            x_train, y_train, xFLAGS.hessian_batch_size,
            lr=xFLAGS.lr, log_dir=xFLAGS.runlogdir,
            grad_measurement=grad_cb)
        callbacks.append(hess_cb)

    if xFLAGS.full_hessian is not None:
        train_batches, test_batches = get_batch_makers(
            xFLAGS.full_hessian_batch_size)
        freq = meas.MeasurementFrequency(
            xFLAGS.full_hessian, xFLAGS.measure_every_step)
        full_hess_cb = meas.FullHessianMeasurement(
            writer, model, freq, train_batches, xFLAGS.runlogdir,
            num_eigenvector_correlations=xFLAGS.output_dim,
            grad_measurement=grad_cb)
        callbacks.append(full_hess_cb)

    if xFLAGS.dataset == 'gaussians':
        freq = meas.MeasurementFrequency(1, xFLAGS.measure_every_step)
        gauss_cb = meas.GaussiansMeasurement(
            writer, model, freq,
            x_train, y_train,
            grad_measurement=grad_cb)
        callbacks.append(gauss_cb)


def save_model(model):
    """Save the model weights."""
    save_dir = xFLAGS.runlogdir + '/saved_models'
    tf.gfile.MakeDirs(save_dir)
    model_path = os.path.join(save_dir, '{}-trained-model.h5'.format(
        run_name()))
    model.save(model_path)
    tf.logging.info('Saved trained model at {} '.format(model_path))


def main(argv):
    del argv  # unused
    if xFLAGS.nothing:
      sys.exit(0)
    init_flags()
    init_randomness()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(xFLAGS.gpu)

    x_train, y_train, x_test, y_test = get_dataset()
    model = get_model(x_train.shape[1:])

    tf.logging.info('Model summary:')
    tf.logging.info(model.summary())

    # initiate optimizer
    if xFLAGS.optimizer_name == 'sgd':
        opt = keras.optimizers.SGD(lr=xFLAGS.lr)
    elif xFLAGS.optimizer_name == 'adam':
        opt = keras.optimizers.adam()
    elif xFLAGS.optimizer_name == 'rmsprop':
        opt = keras.optimizers.rmsprop(lr=xFLAGS.lr, decay=1e-6)
        # opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    elif xFLAGS.optimizer_name == 'projected-gd':
        hessian_spec = tfutils.KerasHessianSpectrum(
            model, x_train, y_train, xFLAGS.projected_batch_size)
        opt = meas.ProjectedGradientDescent(
            xFLAGS.lr, model, x_train, y_train,
            hessian_spec, xFLAGS.projected_num_evs)
    else:
        raise RuntimeError('Unknown optimizer: {}'.format(
            xFLAGS.optimizer_name))

    # Setup logging
    xFLAGS.set('runlogdir',
               '{}/{}'.format(xFLAGS.logdir, run_name()))
    tf.gfile.MakeDirs(xFLAGS.runlogdir)
    log = logging.getLogger('tensorflow')
    log.propagate = False
    sh = logging.StreamHandler(sys.stderr)
    log.addHandler(sh)
    fh = logging.FileHandler('{}/tensorflow.log'.format(xFLAGS.runlogdir))
    log.addHandler(fh)

    tf.logging.info('Started: {}'.format(RUN_TIMESTAMP))
    # full_command = ' '.join(sys.argv)
    # tf.logging.info('Full command:', full_command)
    tf.logging.info('Log dir: {}'.format(xFLAGS.runlogdir))
    tf.logging.info('Run name: {}'.format(run_name()))

    local_device_protos = device_lib.list_local_devices()
    devices = [x.name for x in local_device_protos]
    tf.logging.info("Available devices: {}".format(devices))

    tbutils.save_run_flags(xFLAGS.runlogdir)

    if is_regression():
        model.compile(loss='mean_squared_error',
                      optimizer=opt, metrics=['mae'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt, metrics=['accuracy'])

    tf.logging.info('Total model parameters: {}'.format(
        tfutils.total_num_weights(model)))

    callbacks = []

    if xFLAGS.summaries:
        writer = meas.MeasurementsWriter(
            vars(FLAGS),
            log_dir=xFLAGS.runlogdir)

        add_callbacks(
            callbacks, writer, model,
            x_train, y_train, x_test, y_test)

    tf.logging.info('Training...')

    model.fit(x_train, y_train,
              batch_size=xFLAGS.batch_size,
              epochs=xFLAGS.epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks,
              verbose=xFLAGS.show_progress_bar)

    if xFLAGS.summaries:
        writer.close()
    tf.logging.info('Done training!')

    save_model(model)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=xFLAGS.show_progress_bar)
    tf.logging.info('Test loss: {}'.format(scores[0]))
    tf.logging.info('Test accuracy: {}'.format(scores[1]))


if __name__ == '__main__':
    tf.app.run(main)
