"""TensorFlow utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from time import time
import scope.lanczos as lanczos

KERAS_LEARNING_PHASE_TEST = 0
KERAS_LEARNING_PHASE_TRAIN = 1


class Timer:
  """A simple wallclock timer."""

  def __init__(self):
    self.reset()

  def reset(self):
    self.start = time()

  @property
  def secs(self):
    return time() - self.start


class NumpyPrintEverything:
  """Tell NumPy to print everything.

  Synopsis:

    with NumpyPrintEverything():
        print(numpy_array)
    """

  def __init__(self):
    pass

  def __enter__(self):
    self.saved_threshold = np.get_printoptions()['threshold']
    np.set_printoptions(threshold=np.nan)

  def __exit__(self, type, value, traceback):
    np.set_printoptions(threshold=self.saved_threshold)


class NumpyPrintoptions:
  """Temporarily set NumPy printoptions.

  Synopsis:
    with NumpyPrintoptions(formatter={'float': '{:0.2f}'.format}):
        print(numpy_array)
  """
  def __init__(self, **kwargs):
    self.options = kwargs

  def __enter__(self):
    self.saved_options = np.get_printoptions()
    np.set_printoptions(**self.options)

  def __exit__(self, type, value, traceback):
    np.set_printoptions(**self.saved_options)


class MiniBatchMaker:
  """Shuffle data and split it into batches."""

  def __init__(self, x, y, batch_size):
    assert len(x) == len(y)
    # assert len(x) % batch_size == 0

    self.x = x
    self.y = y

    self.N = len(x)
    self.batch_size = batch_size
    self.steps_per_epoch = \
        (self.N + self.batch_size - 1) // self.batch_size
    self.batches_per_epoch = self.steps_per_epoch

    self.shuffle()

    self.i = 0
    self.epochs_completed = 0
    self.step = 0

  def shuffle(self):
    perm = np.random.permutation(self.N)
    self.shuffled_x = self.x[perm]
    self.shuffled_y = self.y[perm]

  def next_batch(self):
    self.step += 1
    end_idx = min(self.i + self.batch_size, self.N)

    x_batch = self.shuffled_x[range(self.i, end_idx)]
    y_batch = self.shuffled_y[range(self.i, end_idx)]

    self.i = end_idx % self.N

    if self.i == 0:
      self.epochs_completed += 1
      self.shuffle()

    return x_batch, y_batch

  def at_start_of_epoch(self):
    """Are we starting a new epoch?"""
    return self.i == 0


def create_iid_batch_generator(x, y, steps, batch_size, resample_prob=1):
  """Returns an IID mini-batch generator.

  ds = Dataset.from_generator(
    create_iid_batch_generator(x, y, batch_size), ...)

  Args:
    x: Input samples
    y: Labels
    steps: How many steps to run for
    batch_size: Integer size of mini-batch
    resample_prob: Probability of resampling a given sample at each step.
      If a function, the function should return the current resampling 
      probability and will be called every time a batch is generated.
  """
  N = len(x)
  
  def gen():
    samples = np.random.choice(N, batch_size, replace=True)
    
    for step in range(steps):
      yield (x[samples], y[samples])

      try:
        current_resample_prob = resample_prob()
      except TypeError:
        current_resample_prob = resample_prob

      to_replace = np.random.random((batch_size,)) < current_resample_prob
      new_samples = np.random.choice(N, to_replace.sum(), replace=True)
      samples[to_replace] = new_samples

  return gen
  

def _AsList(x):
  return x if isinstance(x, (list, tuple)) else [x]


def keras_feed_dict(model,
                    x=None,
                    y=None,
                    feed_dict={},
                    learning_phase=KERAS_LEARNING_PHASE_TEST):
  """Return a feed dict with inputs and labels suitable for Keras.

    Args:
        model: A Keras Model
        x: Model inputs, or None if inputs are not fed
        y: Model targets (labels), or None if targets are not fed
        feed_dict: Additional feed_dict to merge with (if given, updated in
          place)
        learning_phase: 0 for TEST, 1 for TRAIN

    Returns:
        The new feed_dict (equal to feed_dict if that was provided).
    """
  new_feed_dict = dict(feed_dict)
  if x is not None:
    new_feed_dict[model.inputs[0]] = x
    new_feed_dict[model.sample_weights[0]] = np.ones(x.shape[0])
  if y is not None:
    new_feed_dict[model.targets[0]] = y
  new_feed_dict[K.learning_phase()] = learning_phase  # TEST phase
  return new_feed_dict


def keras_compute_tensors(model, x, y, tensors, feed_dict={}):
  """Compute the given tensors in Keras."""
  new_feed_dict = keras_feed_dict(model, x, y, feed_dict)
  return K.get_session().run(tensors, feed_dict=new_feed_dict)


def clone_keras_model_shared_weights(
    model, input_tensor, target_tensor):
  """Clone a Keras model.
  
  The new model shares its weights with the old model, but accepts different
  inputs and targets. This is useful, for example, for evaluating a model
  mid-training.

  Args:
      model: A compiled Keras model.
      input_tensor: Tensor to use as input for the cloned model.
      target_tensor: Tensor to be used as targets (labels) for the cloned model.

  Returns:
      The cloned Keras model.
  """
  assert len(model.inputs) == 1
  inputs = keras.layers.Input(tensor=input_tensor,
                              shape=model.inputs[0].shape[1:])
  clone = keras.Model(
    inputs=inputs,
    outputs=model(input_tensor))
  clone.compile(
      loss=model.loss,
      target_tensors=[target_tensor],
      optimizer=model.optimizer,
      metrics=model.metrics)
  return clone


def flatten_array_list(arrays):
  """Flatten and concat a list of numpy arrays into a single rank 1 vector."""
  return np.concatenate([np.reshape(a, [-1]) for a in arrays], axis=0)


def flatten_tensor_list(tensors):
  """Flatten and concat a list of tensors into a single rank 1 tensor."""
  return tf.concat([tf.reshape(t, [-1]) for t in tensors], axis=0)


def unflatten_tensor_list(flat_tensor, orig_tensors):
  """Reshape a flattened tensor back to a list of tensors with their
  original shapes.
  
  Args:
    flat_tensor: A tensor that was previously flattened using
      flatten_tensor_list()
    orig_tensor: A list of tensors with the original desired shapes.
  """
  unflattened = []
  offset = 0
  for t in orig_tensors:
    num_elems = t.shape.num_elements()
    unflattened.append(
        tf.reshape(flat_tensor[offset:offset + num_elems], t.shape))
    offset += num_elems
  return unflattened


def compute_sample_mean_tensor(model, batches, tensors, feed_dict={}):
  """Compute the sample mean of the given tensors.

  Args:
    model: Keras Model
    batches: MiniBatchMaker
    tensors: Tensor or list of Tensors to compute the mean of
    feed_dict: Used when evaluating tensors
  """
  sample_means = None
  tensors_is_list = isinstance(tensors, (list, tuple))
  tensors = _AsList(tensors)

  while True:
    x_batch, y_batch = batches.next_batch()
    results = keras_compute_tensors(model, x_batch, y_batch, tensors, feed_dict)

    for i in range(len(results)):
      results[i] *= len(x_batch)

    if sample_means is None:
      sample_means = results
    else:
      for i in range(len(results)):
        sample_means[i] += results[i]
    if batches.at_start_of_epoch():
      break

  for i in range(len(sample_means)):
    sample_means[i] /= batches.N

  if tensors_is_list:
    return sample_means
  else:
    assert len(sample_means) == 1
    return sample_means[0]


def jacobian(y, x):
  """Compute the Jacobian tensor J_ij = dy_i/dx_j.

    From https://github.com/tensorflow/tensorflow/issues/675, which is adapted
    from tf.hessiangs().

    :param Tensor y: A Tensor
    :param Tensor x: A Tensor
    :rtype: Tensor
    :return: The Jacobian Tensor, whose shape is the concatenation of
    the y_flat and x shapes.
    """
  y_flat = tf.reshape(y, [-1])
  # tf.shape() returns a Tensor, so this supports dynamic sizing
  n = tf.shape(y_flat)[0]

  loop_vars = [
      tf.constant(0, tf.int32),
      tf.TensorArray(tf.float32, size=n),
  ]

  _, jacobian = tf.while_loop(
      lambda j, _: j < n,
      lambda j, result: (j+1,
                         result.write(j, tf.gradients(y_flat[j], x)[0])),
      loop_vars)

  jacobian_shape = tf.concat([tf.shape(y), tf.shape(x)], axis=0)
  jacobian = tf.reshape(jacobian.stack(), jacobian_shape)

  return jacobian


def jacobians(y, xs):
  """Compute the Jacobian tensors J_ij = dy_i/dx_j for each x in xs.

    With this implementation, the gradient is computed for all xs in one
    call, so if xs includes weights from different layers then back prop
    is used.

    :param Tensor y: A rank 1 Tensor
    :param Tensor xs: A Tensor or list of Tensors
    :rtype: list
    :return: List of Jacobian tensors J_ij = dy_i/dx_j for each x in xs.
    """
  if y.shape.ndims != 1:
    raise ValueError('y must be a rank 1 Tensor')
  xs = _AsList(xs)
  # tf.shape() returns a Tensor, so this supports dynamic sizing
  len_y = tf.shape(y)[0]
  jacobians = []

  # Outer loop runs over elements of y, computes gradients for each
  loop_vars = [
      tf.constant(0, tf.int32),
      [tf.TensorArray(tf.float32, size=len_y) for x in xs]
  ]

  def _compute_single_y_gradient(j, arrays):
    """Compute the gradient for a single y elem."""
    grads = tf.gradients(y[j], xs)
    for i, g in enumerate(grads):
      arrays[i] = arrays[i].write(j, g)
    return arrays

  _, jacobians = tf.while_loop(
      lambda j, _: j < len_y,
      lambda j, arrays: (j + 1, _compute_single_y_gradient(j, arrays)),
      loop_vars)

  jacobians = [a.stack() for a in jacobians]
  return jacobians


def hessians(y, xs):
  """The Hessian of y with respect to each x in xs.

    :param y Tensor: A scalar Tensor.
    :param xs Tensor: A Tensor or list of Tensors. Each Tensor can have any
    rank.
    :rtype: list
    :return: List of Hessians d^2y/dx^2. The shape of a Hessian is
    x.shape + y.shape.
    """
  xs = _AsList(xs)
  hessians = []

  for x in xs:
    # First derivative and flatten
    grad = tf.gradients(y, x)[0]
    grad_flat = tf.reshape(grad, [-1])

    # Second derivative
    n = tf.shape(grad_flat)[0]
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]

    _, hessian = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j+1, result.write(
            j, tf.gradients(grad_flat[j], x)[0])),
        loop_vars)

    hessian = hessian.stack()

    x_shape = tf.shape(x)
    hessian_shape = tf.concat([x_shape, x_shape], axis=0)
    hessians.append(tf.reshape(hessian, hessian_shape))

  return hessians


def num_weights(weights):
  """Number of weights in the given list of weight tensors."""
  return sum([w.shape.num_elements() for w in weights])


def total_num_weights(model):
  """Total number of weights in the given Keras model."""
  return num_weights(model.trainable_weights)


def total_tensor_elements(x):
  """Tensor containing the total number of elements of x.

    :param x Tensor: A tensor.
    :rtype: Tensor
    :return: A scalar Tensor containing the total number of elements.
    """
  return tf.reduce_prod(tf.shape(x))


def hessian_tensor_blocks(y, xs):
  """Compute the tensors that make up the full Hessian (d^2y / dxs dxs).

    A full computation of the Hessian would look like this:

    blocks = hessian_tensor_blocks(y, xs)
    block_results = sess.run(blocks)
    hessian = hessian_combine_blocks(block_results)

    :param y Tensor: A scalar Tensor.
    :param xs Tensor: A Tensor or list of Tensors. Each Tensor can have any
    rank.
    :rtype: list
    :return: List of Tensors that should be evaluated, and the results
    should be passed to hessian_combine_blocks() to get the full
    gradient.
    """
  xs = _AsList(xs)
  hess_blocks = []

  for i1, x1 in enumerate(xs):
    # First derivative and flatten
    grad_x1 = tf.gradients(y, x1)[0]
    grad_x1_flat = tf.reshape(grad_x1, [-1])
    x1_size = total_tensor_elements(x1)

    # Second derivative: Only compute upper-triangular blocks
    # because Hessian is symmetric
    for x2 in xs[i1:]:
      x2_size = total_tensor_elements(x2)
      loop_vars = [
          tf.constant(0, tf.int32),
          tf.TensorArray(tf.float32, size=x1_size),
      ]

      _, x1_x2_block = tf.while_loop(
          lambda j, _: j < x1_size,
          lambda j, result: (j+1, result.write(
              j, tf.gradients(grad_x1_flat[j], x2)[0])),
          loop_vars)

      x1_x2_block = tf.reshape(x1_x2_block.stack(), [x1_size, x2_size])
      hess_blocks.append(x1_x2_block)

  return hess_blocks


def hessian_combine_blocks(blocks):
  """Combine the pieces obtained by evaluated the Tensors returned

    by hessian_tensor_pieces(), and return the full Hessian matrix.
    """
  # We only record upper-triangular blocks, and here we work out
  # the number of blocks per row by solving a quadratic:
  # len = n * (n+1) / 2
  num_recorded_blocks = len(blocks)
  blocks_per_row = int((np.sqrt(1 + 8 * num_recorded_blocks) - 1) / 2)

  # Sum the column sizes
  dims = [b.shape[1] for b in blocks[:blocks_per_row]]
  total_dim = sum(dims)

  H = np.zeros((total_dim, total_dim))
  row = 0
  col = 0

  for i, b in enumerate(blocks):
    row_start = sum(dims[:row])
    row_end = sum(dims[:row + 1])

    col_start = sum(dims[:col])
    col_end = sum(dims[:col + 1])

    H[row_start:row_end, col_start:col_end] = b
    H[col_start:col_end, row_start:row_end] = b.transpose()

    col += 1
    if col >= blocks_per_row:
      row += 1
      col = row

  return H


def trace_hessian(loss, logits, weights):
  """Compute the trace of the Hessian of loss with respect to weights.

    We assume that loss = loss(logits(weights)), and that logits is a
    piecewise-linear function of the weights (therefore d^2 logits / dw^2 = 0
    for any w). This allows for a faster implementation that the naive one.

    Note: This computes the Hessian of loss / logits, where logits is indexed
    by sample and class, but all elements of the Hessian with two different
    classes vanish. So this is still not the most efficient way to do it.

    :param loss Tensor: A scalar Tensor
    :param logits Tensor: The logits tensor.
    :param weights Tensor: A Tensor or list of Tensors of model weights.
    :rtype: Tensor
    :return: The trace of the Hessian of loss with respect to all the weights.
    """
  weights = _AsList(weights)

  # Flatten logits with a well-specified dimension. This assumes any
  # non-trivial dimension will resolve to 1.
  # logits_flat = tf.reshape(logits, [-1])

  loss_logits_hessian = hessians(loss, logits)[0]
  tr_hessian_pieces = []

  for w in weights:
    J = jacobian(logits, w)
    # Contract along the weight indices
    # (first index is the logit index)
    weight_axes = list(range(logits.shape.ndims, J.shape.ndims))
    JJ = tf.tensordot(J, J, axes=[weight_axes, weight_axes])
    # Doesn't work for dynamic shape
    # assert loss_logits_hessian.shape == JJ.shape
    all_axes = list(range(JJ.shape.ndims))
    tr_hessian_pieces.append(
        tf.tensordot(loss_logits_hessian, JJ, axes=[all_axes, all_axes]))

  return tf.reduce_sum(tr_hessian_pieces)


def trace_hessian_softmax_crossentropy(logits, weights):
  """Compute the trace of the Hessian of loss with respect to weights.

    The loss is assumed to be crossentropy(softmax(logits)), which allows
    us to compute the loss/logits Hessian analytically, and it factorizes.

    We also assume that logits is a piecewise-linear function of the weights
    (therefore d^2 logits / dw^2 = 0 for any w).

    Here we compute the Hessian of loss / logits analytically, which saves
    some time, but only about 10%. It seems most time is spent just computing
    the Jacobian.

    :param loss Tensor: A scalar Tensor
    :param logits Tensor: A Tensor with rank 2, indexed by [sample, class].
    :param weights Tensor: A Tensor or list of Tensors of model weights.
    :rtype: Tensor
    :return: The trace of the Hessian of loss with respect to all the weights.
    """
  weights = _AsList(weights)

  if logits.shape.ndims != 2:
    raise ValueError('logits tensor must have rank 2')

  probs = tf.nn.softmax(logits)
  tr_hessian_pieces = []

  JdotP = tf.gradients(logits, weights, grad_ys=probs)
  tf.logging.info('JdotP =', JdotP)

  return tf.reduce_sum(tr_hessian_pieces)


def trace_hessian_reference(loss, weights):
  """Compute the whole Hessian for each layer, then take the trace.

    This is a straightforward and slow implementation meant for testing.
  """
  weights = _AsList(weights)
  trace_terms = []
  grads = tf.gradients(loss, weights)

  for grad, weight in zip(grads, weights):
    grad_unstacked = tf.unstack(tf.reshape(grad, [-1]))
    for i, g in enumerate(grad_unstacked):
      g2 = tf.reshape(tf.gradients(g, weight)[0], [-1])
      diag_hessian_term = g2[i]
      trace_terms.append(diag_hessian_term)

  return tf.reduce_sum(trace_terms)


def hessian_vector_product(loss, weights, v):
  """Compute the tensor of the product H.v, where H is the loss Hessian with

    respect to the weights. v is a vector (a rank 1 Tensor) of the same size as
    the loss gradient. The ordering of elements in v is the same obtained from
    flatten_tensor_list() acting on the gradient. Derivatives of dv/dweights
    should vanish.
    """
  grad = flatten_tensor_list(tf.gradients(loss, weights))
  grad_v = tf.reduce_sum(grad * tf.stop_gradient(v))
  H_v = flatten_tensor_list(tf.gradients(grad_v, weights))
  return H_v


class TensorStatistics:
  """Collect statistics for a tensor over different mini-batches."""

  def __init__(self, tensor):
    self.tensor = tensor
    shape = tensor.shape.as_list()
    self.running_sum = np.zeros(shape, dtype=np.float32)
    self.running_sum_of_squares = np.zeros(shape, dtype=np.float32)
    self.n = 0

  def add_minibatch(self, value):
    """Add mean value over minibatch."""
    self.running_sum += value
    self.running_sum_of_squares += value * value
    self.n += 1

  @property
  def mean(self):
    """The mean"""
    return self.running_sum / self.n

  @property
  def var(self):
    """Variance of each tensor element"""
    return self.running_sum_of_squares / self.n - self.mean**2

  @property
  def std(self):
    """Standard deviation of each tensor element"""
    return np.sqrt(self.var)

  @property
  def norm_of_mean(self):
    """Norm of the mean"""
    return np.linalg.norm(self.mean)

  @property
  def norm_of_std(self):
    """Norm of vector of standard deviations"""
    return np.linalg.norm(self.std)


class TensorListStatistics(list):
  """Collect statistics for a list of tensors over different mini-batches.

    Behaves as list where each element is a TensorStatistics object.
  """

  def __init__(self, tensors):
    """tensors: list of Tensors"""
    super().__init__([TensorStatistics(t) for t in tensors])

  def add_minibatch(self, values):
    for stat, val in zip(self, values):
      stat.add_minibatch(val)

  @property
  def means(self):
    """List of tensor means"""
    return [s.mean for s in self]

  @property
  def vars(self):
    """List of tensor variances"""
    return [s.var for s in self]

  @property
  def stds(self):
    """List of tensor standard devs"""
    return [s.std for s in self]

  @property
  def norm_of_mean(self):
    """The norm of the concatenated list of tensor means."""
    norms = np.array([np.linalg.norm(s.mean) for s in self])
    return np.sqrt(np.sum(norms * norms))

  @property
  def norm_of_std(self):
    """The norm of the concatenated list of tensor stds."""
    norms = np.array([np.linalg.norm(s.std) for s in self])
    return np.sqrt(np.sum(norms * norms))


class KerasHessianSpectrum:
  """Computes the partial Hessian spectrum of a Keras model using Lanczos."""

  def __init__(self, model, x, y, batch_size=1024, weights=None, loss=None):
    """model is a keras sequential model.

    Args:
        model: A Keras Model
        x: Training samples
        y: Training labels
        batch_size: Batch size for computing the Hessian (affects performance
            but not results)
        weights: Weights with respect to which to compute the Hessian.
            Can be a weight tensor or a list of tensors. If None,
            all model weights are used.
        loss: Can be specified separately for unit testing purposes.
    """
    self.model = model

    if weights is None:
      self.weights = model.trainable_weights
    else:
      self.weights = _AsList(weights)

    self.num_weights = num_weights(self.weights)
    self.v = tf.placeholder(tf.float32, shape=(self.num_weights,))

    # Delay looking at model because it may not be compiled yet
    self._loss = loss
    self._Hv = None
    self.train_batches = MiniBatchMaker(x, y, batch_size)

  @property
  def loss(self):
    """The loss.

    Evaluated lazily in case the model is not compiled at
        first.
    """
    if self._loss is None:
      return self.model.total_loss
    else:
      return self._loss

  @property
  def Hv(self):
    """The Hessian-vector product tensor"""
    if self._Hv is None:
      self._Hv = hessian_vector_product(self.loss, self.weights, self.v)
    return self._Hv

  def compute_spectrum(self, k, show_progress=False):
    """Compute k leading eigenvalues and eigenvectors."""
    self.lanczos_iterations = 0
    timer = Timer()
    evals, evecs = lanczos.eigsh(self.num_weights, np.float32,
                                 lambda v: self._compute_Hv(v, show_progress),
                                 k)
    self.lanczos_secs = timer.secs
    if show_progress:
      tf.logging.info('')
    return evals, evecs

  def compute_other_edge(self,
                         leading_ev,
                         session=None,
                         epsilon=1e-3,
                         min_iters=0,
                         max_iters=1000,
                         matrix_vector_action=None,
                         debug=False):
    """DEPRECATED.

    Given the leading eigenvalue leading_ev, compute the
        eigenvalue at the opposite edge of the spectrum.
    """
    self.lanczos_iterations = 0
    other_edge_shifted, evec = self.compute_leading_ev(
        session,
        epsilon,
        min_iters,
        max_iters,
        matrix_vector_action=lambda v, Hv: Hv - leading_ev * v,
        debug=debug)
    other_edge = other_edge_shifted + leading_ev
    return other_edge, evec

  def compute_leading_ev(self,
                         session=None,
                         epsilon=1e-3,
                         min_iters=0,
                         max_iters=1000,
                         matrix_vector_action=None,
                         debug=False):
    """DEPRECATED.

    epsilon is the tolerance when deciding when the
        power-law method converges. matrix_vector_action is a function that
        takes (v, Hv) and returns a vector that represents the action v -> A.v
        of some linear operator A, which may depend on H. If
        matrix_vector_action is specified, The leading eigenvalue of A will be
        computed instead of H.
    """
    self.lanczos_iterations = 0
    v = np.random.randn(self.num_weights).astype(np.float32)
    v = v / np.linalg.norm(v)
    i = 0
    tf.logging.debug('In compute_leading_ev ..')

    # The default matrix-vector action to use
    def Hv_action(v, Hv):
      return Hv

    if matrix_vector_action is None:
      matrix_vector_action = Hv_action

    while i < max_iters:
      start = time()
      # Act twice so we don't care about the eigenvalue sign
      Hv = matrix_vector_action(v, self._compute_Hv(v, False))
      H2v = matrix_vector_action(Hv, self._compute_Hv(Hv, False))
      H2v_normal = H2v / np.linalg.norm(H2v)
      # This gives the signed eigenvalue.
      # v is a unit vector so we don't need to normalize
      maybe_ev = v.dot(Hv)
      if debug:
        elapsed = time() - start
        tf.logging.dbeug('iter =', i, ' |v-H^2v/|H^2v|| =',
                         np.linalg.norm(H2v_normal - v), ' maybe_ev =',
                         maybe_ev, ' ({} seconds)'.format(elapsed))
      if np.linalg.norm(H2v_normal - v) < epsilon and i >= min_iters:
        break
      v = H2v_normal
      i += 2

    if i == max_iters:
      raise ValueError(
          'Max iteration {} reached without convergence'.format(max_iters))

    return maybe_ev, v

  def _compute_Hv(self, v, show_progress):
    if show_progress:
      print('.', end='')
      sys.stdout.flush()
    self.lanczos_iterations += 1
    return compute_sample_mean_tensor(self.model, self.train_batches, self.Hv,
                                      {self.v: v})


#############
# Old code
#############

# TODO keep only the 2-pt function code
# class GradientStatistics:
#     '''Collect gradient statistics from minibatch gradients, including
#     norm of gradient mean and std.'''
#     def __init__(self, name, compute_2pt):
#         '''name is used for debugging.'''
#         self.name = name
#         self.compute_2pt = compute_2pt

#         self.grad_accum = MeanCalculator()
#         self.grad_sqr_accum = MeanCalculator()
#         self.grad_2pt_accum = MeanCalculator()

#     def add_minibatch_gradient(self, grad, grad_term1):
#         grad_sqr = grad * grad
#         grad_2pt = self._compute_2pt(grad)

#         self.grad_accum.add(grad)
#         self.grad_sqr_accum.add(grad_sqr)

#         if grad_2pt is not None:
#             self.grad_2pt_accum.add(grad_2pt)

#     def compute_statistics(self):
#         '''Gradient mean norm and std per layer, averaging over batches. n is
#         the number of elements in each sum.'''
#         mean_grad = self.grad_accum.mean
#         mean_grad_norm = np.linalg.norm(mean_grad)
#         mean_grad_sqr = self.grad_sqr_accum.mean
#         std_grad = np.sqrt(mean_grad_sqr - mean_grad * mean_grad)
#         self.mean = mean_grad_norm
#         self.std = np.linalg.norm(std_grad)

#         if self.has_2pt:
#             mean_grad_2pt = self.grad_2pt_accum.mean

#             # The noise is given by the connected 2-point function
#             grad_2pt_connected = mean_grad_2pt - self._compute_2pt(mean_grad)

#             # V columns are eigenvectors. eigenvalues are in ascending order.
#             # self.eig_2pt, V = np.linalg.eigh(grad_2pt_connected)
#             num_evs = 10
#             self.eig_2pt, V = scipy.linalg.eigh(
#                 grad_2pt_connected,
#                 eigvals=(len(grad_2pt_connected) - num_evs,
#                          len(grad_2pt_connected) - 1))
#             flat_grad = mean_grad.flatten()

#             # Gradient expanded in noise eigenbasis
#             self.grad_in_noise_basis = V.conj().T.dot(
#                 flat_grad / mean_grad_norm)
#             # print('noise evs =', self.eig_2pt[-10:])
#             # print('grad_in_noise_basis =',
#             #       self.grad_in_noise_basis[-10:])

#             # This is grad^T . 2pt . grad / |grad|^2
#             expectation = flat_grad.dot(grad_2pt_connected.dot(flat_grad))
#             self.grad_on_2pt = expectation / mean_grad_norm**2

#     @property
#     def has_2pt(self):
#         return self.grad_2pt_accum.n > 0

#     def _compute_2pt(self, g):
#         '''Given a vector g, compute the matrix A_ij = g_i g_j'''
#         # Can't compute 2pt if the size d is too big, because the
#         # 2pt has size d^2, and diagonalization also take a while.
#         # print('{} ::: g.size = {}'.format(self.name, g.size))
#         if not self.compute_2pt:
#             return None
#         if g.size > 5000:
#             return None
#         g_flat = g.flatten()
#         g_row = g_flat.reshape((1, len(g_flat)))
#         g_col = g_flat.reshape((len(g_flat), 1))
#         g_mat = g_col.dot(g_row)
#         return g_mat
