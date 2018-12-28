"""A set of simple deep learning models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization


def _dense_layer(cmd_args, *args, **kwargs):
  """Create a Dense layer with biases and regularization as specified by

    cmd_args.
  """
  kwargs['use_bias'] = cmd_args.use_bias
  kwargs['kernel_regularizer'] = cmd_args.l2_regularizer
  return Dense(*args, **kwargs)


def _conv2d_layer(cmd_args, *args, **kwargs):
  """Create a Conv2D layer with biases and regularization as specified by

    cmd_args.
  """
  kwargs['use_bias'] = cmd_args.use_bias
  kwargs['kernel_regularizer'] = cmd_args.l2_regularizer
  return Conv2D(*args, **kwargs)


def add_dense_layers(args, model, widths, input_shape=None):
  """Add dense layers to the model.

    Args:
        args: Arguments that determine the activation, dropout, and batch-norm.
        model: The Keras model to which to add layers.
        widths: List of integer layer widhts.
        input_shape: Shape of input layer.
  """
  for width in widths:
    try:
      model.add(_dense_layer(args, width))
    except ValueError:
      # If it's the first layer we need to specify the input shape
      model.add(_dense_layer(args, width, input_shape=input_shape))
    if args.batch_norm:
      model.add(BatchNormalization())
    model.add(Activation(args.activation))
    if args.dropout:
      model.add(Dropout(0.25))


def regression_fc_model(args, output_dim):
  """Fully-connected regression"""
  model = keras.models.Sequential()
  if not args.fc_widths:
    model.add(
        _dense_layer(args, output_dim, name='regression_map', input_shape=(1,)))
  else:
    add_dense_layers(args, model, args.fc_widths, input_shape=(1,))
    model.add(_dense_layer(args, output_dim, name='regression_map'))
  return model


def classification_fc_model(args, input_shape, num_classes):
  """Fully-connected classification"""
  model = keras.models.Sequential()
  model.add(keras.layers.InputLayer(input_shape=input_shape))

  if len(input_shape) >  1:
    model.add(Flatten())

  add_dense_layers(args, model, args.fc_widths)

  try:
    logits = _dense_layer(args, num_classes, name='logits')
    model.add(logits)
  except ValueError:
    logits = _dense_layer(
        args, num_classes, name='logits')
    model.add(logits)

  model.add(Activation('softmax', name='softmax'))
  return model


def classification_convnet_model(args, input_shape, num_classes):
  """Convnet classification"""
  model = keras.models.Sequential()
  model.add(keras.layers.InputLayer(input_shape=input_shape))
  model.add(_conv2d_layer(args, 32, (3, 3), padding='same'))
  if args.batch_norm:
    # The default Conv2D data format is 'channels_last' and the
    # default BatchNormalization axes is -1.
    #
    # In batch norm fix a channel and average over the samples
    # and the spatial location. axis = the axis we keep fixed
    # (averaging over the rest), so for 'channels_last' this is
    # the last axis, -1.
    model.add(BatchNormalization())
  model.add(Activation(args.activation))
  model.add(_conv2d_layer(args, 32, (3, 3)))
  if args.batch_norm:
    model.add(BatchNormalization())
  model.add(Activation(args.activation))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  if args.dropout:
    model.add(Dropout(0.25))

  model.add(_conv2d_layer(args, 64, (3, 3), padding='same'))

  if args.batch_norm:
    model.add(BatchNormalization())
  model.add(Activation(args.activation))
  model.add(_conv2d_layer(args, 64, (3, 3)))
  if args.batch_norm:
    model.add(BatchNormalization())
  model.add(Activation(args.activation))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  if args.dropout:
    model.add(Dropout(0.25))

  model.add(Flatten())
  if args.dense:
    # model.add(Dense(512, name='dense'))
    model.add(_dense_layer(args, args.cnn_last_layer, name='dense'))
    if args.overparam > 0:
      for i in range(args.overparam):
        model.add(
            _dense_layer(
                args,
                args.cnn_last_layer,
                name='dense-overparam{}'.format(i + 1)))
    if args.batch_norm:
      model.add(BatchNormalization())
    model.add(Activation(args.activation))
    if args.dropout:
      model.add(Dropout(0.5))
  if args.overparam > 0:
    for i in range(args.overparam):
      model.add(
          _dense_layer(
              args, num_classes, name='output-overparam{}'.format(i + 1)))
  logits = _dense_layer(args, num_classes, name='logits')
  model.add(logits)
  model.add(Activation('softmax'))
  return model


def classification_small_convnet_model(args, input_shape, num_classes):
  """Convnet classification"""
  model = keras.models.Sequential()
  model.add(keras.layers.InputLayer(input_shape=input_shape))
  model.add(
      _conv2d_layer(args, 32, (3, 3), padding='same'))
  if args.batch_norm:
    # The default Conv2D data format is 'channels_last' and the
    # default BatchNormalization axes is -1.
    #
    # In batch norm fix a channel and average over the samples
    # and the spatial location. axis = the axis we keep fixed
    # (averaging over the rest), so for 'channels_last' this is
    # the last axis, -1.
    model.add(BatchNormalization())
  model.add(Activation(args.activation))
  model.add(_conv2d_layer(args, 32, (3, 3)))
  if args.batch_norm:
    model.add(BatchNormalization())
  model.add(Activation(args.activation))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  if args.dropout:
    model.add(Dropout(0.25))

  model.add(_conv2d_layer(args, 64, (3, 3), padding='same'))
  if args.batch_norm:
    model.add(BatchNormalization())
  model.add(Activation(args.activation))
  model.add(_conv2d_layer(args, 64, (3, 3)))
  if args.batch_norm:
    model.add(BatchNormalization())
  model.add(Activation(args.activation))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  if args.dropout:
    model.add(Dropout(0.25))

  model.add(Flatten())
  # if args.dense:
  #     model.add(Dense(args.cnn_last_layer, name='dense'))
  #     if args.overparam > 0:
  #         for i in range(args.overparam):
  #             model.add(Dense(args.cnn_last_layer,
  #                             name='dense-overparam{}'.format(i + 1)))
  #     if args.batch_norm:
  #         model.add(BatchNormalization())
  #     model.add(Activation(args.activation))
  #     if args.dropout:
  #         model.add(Dropout(0.5))
  if args.overparam > 0:
    for i in range(args.overparam):
      model.add(
          _dense_layer(
              args, num_classes, name='output-overparam{}'.format(i + 1)))
  logits = _dense_layer(args, num_classes, name='logits')
  model.add(logits)
  model.add(Activation('softmax'))
  return model
