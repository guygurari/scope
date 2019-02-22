"""A set of simple deep learning models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow
import tensorflow.keras as keras

from tensorflow.python.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.python.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.python.keras.regularizers import l2

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Sequential

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
    ### Aitor: changed order of batchnorm, because I want to normalize the activations
    model.add(Activation(args.activation))

    if args.batch_norm:
      if args.mean_to_zero:
        bconstraint = keras.constraints.MaxNorm(max_value=0)
      else:
        bconstraint=None
      model.add(BatchNormalization(beta_constraint=bconstraint))
    #model.add(Activation(args.activation))
    if args.dropout:
      model.add(Dropout(0.25))


def regression_fc_model(args, output_dim):
  """Fully-connected regression"""
  model = Sequential()
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
  # if args.batch_norm:
  #   # The default Conv2D data format is 'channels_last' and the
  #   # default BatchNormalization axes is -1.
  #   #
  #   # In batch norm fix a channel and average over the samples
  #   # and the spatial location. axis = the axis we keep fixed
  #   # (averaging over the rest), so for 'channels_last' this is
  #   # the last axis, -1.
  #   model.add(BatchNormalization())
  # model.add(Activation(args.activation))

  ## Aitor
  model.add(Activation(args.activation))
  if args.batch_norm:
    if args.mean_to_zero:
      bconstraint=keras.constraints.MaxNorm(max_value=0)
    else:
      bconstraint=None
    model.add(BatchNormalization(beta_constraint=bconstraint))
  ## Aitor
  model.add(_conv2d_layer(args, 32, (3, 3)))
  # if args.batch_norm:
  #   model.add(BatchNormalization())
  # model.add(Activation(args.activation))
   ## Aitor
  model.add(Activation(args.activation))
  if args.batch_norm:
    if args.mean_to_zero:
      bconstraint=keras.constraints.MaxNorm(max_value=0)
    else:
      bconstraint=None
    model.add(BatchNormalization(beta_constraint=bconstraint))
  ## Aitor
  model.add(MaxPooling2D(pool_size=(2, 2)))
  if args.dropout:
    model.add(Dropout(0.25))

  model.add(_conv2d_layer(args, 64, (3, 3), padding='same'))

  # if args.batch_norm:
  #   model.add(BatchNormalization())
  # model.add(Activation(args.activation))
   ## Aitor
  model.add(Activation(args.activation))
  if args.batch_norm:
    if args.mean_to_zero:
      bconstraint=keras.constraints.MaxNorm(max_value=0)
    else:
      bconstraint=None
    model.add(BatchNormalization(beta_constraint=bconstraint))
  ## Aitor
  model.add(_conv2d_layer(args, 64, (3, 3)))
  # if args.batch_norm:
  #   model.add(BatchNormalization())
  # model.add(Activation(args.activation))
   ## Aitor
  model.add(Activation(args.activation))
  if args.batch_norm:
    if args.mean_to_zero:
      bconstraint=keras.constraints.MaxNorm(max_value=0)
    else:
      bconstraint=None
    model.add(BatchNormalization(beta_constraint=bconstraint))
  ## Aitor
  model.add(MaxPooling2D(pool_size=(2, 2)))
  if args.dropout:
    model.add(Dropout(0.25))

  model.add(Flatten())

  if args.dense:
    model.add(_dense_layer(args, args.cnn_last_layer, name='dense'))
    if args.overparam > 0:
      for i in range(args.overparam):
        model.add(
            _dense_layer(
                args,
                args.cnn_last_layer,
                name='dense-overparam{}'.format(i + 1)))
    # if args.batch_norm:
    #   model.add(BatchNormalization())
    # model.add(Activation(args.activation))
     ## Aitor
  model.add(Activation(args.activation))
  if args.batch_norm:
    if args.mean_to_zero:
      bconstraint=keras.constraints.MaxNorm(max_value=0)
    else:
      bconstraint=None
    model.add(BatchNormalization(beta_constraint=bconstraint))
  ## Aitor
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
  # if args.batch_norm:
    # The default Conv2D data format is 'channels_last' and the
    # default BatchNormalization axes is -1.
    #
    # In batch norm fix a channel and average over the samples
    # and the spatial location. axis = the axis we keep fixed
    # (averaging over the rest), so for 'channels_last' this is
    # the last axis, -1.
  #   model.add(BatchNormalization())
  # model.add(Activation(args.activation))
   ## Aitor
  model.add(Activation(args.activation))
  if args.batch_norm:
    if args.mean_to_zero:
      bconstraint=keras.constraints.MaxNorm(max_value=0)
    else:
      bconstraint=None
    model.add(BatchNormalization(beta_constraint=bconstraint))
  ## Aitor
  model.add(_conv2d_layer(args, 32, (3, 3)))
  # if args.batch_norm:
  #   model.add(BatchNormalization())
  # model.add(Activation(args.activation))
   ## Aitor
  model.add(Activation(args.activation))
  if args.batch_norm:
    if args.mean_to_zero:
      bconstraint=keras.constraints.MaxNorm(max_value=0)
    else:
      bconstraint=None
    model.add(BatchNormalization(beta_constraint=bconstraint))
  ## Aitor
  model.add(MaxPooling2D(pool_size=(2, 2)))
  if args.dropout:
    model.add(Dropout(0.25))

  model.add(_conv2d_layer(args, 64, (3, 3), padding='same'))
  # if args.batch_norm:
  #   model.add(BatchNormalization())
  # model.add(Activation(args.activation))
   ## Aitor
  model.add(Activation(args.activation))
  if args.batch_norm:
    if args.mean_to_zero:
      bconstraint=keras.constraints.MaxNorm(max_value=0)
    else:
      bconstraint=None
    model.add(BatchNormalization(beta_constraint=bconstraint))
  ## Aitor
  model.add(_conv2d_layer(args, 64, (3, 3)))
  # if args.batch_norm:
  #   model.add(BatchNormalization())
  # model.add(Activation(args.activation))
   ## Aitor
  model.add(Activation(args.activation))
  if args.batch_norm:
    if args.mean_to_zero:
      bconstraint=keras.constraints.MaxNorm(max_value=0)
    else:
      bconstraint=None
    model.add(BatchNormalization(beta_constraint=bconstraint))
  ## Aitor
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

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 fused_batch_norm=False):


    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
        fused_batch_norm (bool): whether to use fused implementation of
            bn (leads to infinite loop with Hessian-vector products).
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization(fused=fused_batch_norm)(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization(fused=fused_batch_norm)(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v2(input_shape, depth, num_classes=10, fused_batch_norm=False):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True,
                     fused_batch_norm=fused_batch_norm)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization(fused=fused_batch_norm)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
