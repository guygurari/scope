"""Implement learning rate schedules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

from absl import flags

import scope.measurements as meas

FLAGS = flags.FLAGS


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


# Inherits from Measurement so it automatically keeps track of the step
# TODO would be nicer to pass it a time-keeping object
class LearningRateSchedule(meas.Measurement):
  """Base class for learning rate schedules."""
  def __init__(self, lr_tensor, recorder):
    """Ctor.

    Args:
      lr_tensor: Tensor holding the learning rate during training
      recorder: MeasurementsRecorder that records the overlap.
        If None, set_recorder() should be called before training.
    """
    super(LearningRateSchedule, self).__init__(
        meas.Frequency(freq=1, stepwise=True),
        recorder=None)
    self.lr_tensor = lr_tensor
    self.recorder = recorder

  def set_recorder(self, recorder):
    self.recorder = recorder

  @abc.abstractmethod
  def lr(self):
    """Returns the current learning rate.

    Can use self.step to get the current training step."""
    pass

  def feed_dict(self):
    """Returns a feed_dict with the learning rate filled in."""
    return {self.lr_tensor: self.lr()}


class LinearDecaySchedule(LearningRateSchedule):
  """Update the learning rate according to a linear decay schedule,
  and record it. Used for example in 1811.03600."""
  def __init__(self, lr_tensor, lr0, alpha=None, T=None):
    """Ctor. The learning rate at step t will be given by:

    lr(t) = lr0 - (1-alpha) * lr0 * t / T   if t <= T
    lr(t) = alpha * lr0                      if t > T

    Args:
      lr_tensor: Tensor holding the learning rate during training
      lr0: Initial learning rate
      alpha: Linear decay coefficient, or None to keep constant lr
      T: Time at which to stop decaying, or None to keep constant lr
    """
    super(LinearDecaySchedule, self).__init__(lr_tensor, recorder=None)
    self.lr0 = lr0
    self.alpha = alpha
    self.T = T

  def lr(self):
    """Returns the current learning rate"""
    if self.T is None:
      return self.lr0
    else:
      return linear_decay(self.lr0, self.alpha, self.T, self.step)


class OverlapSchedule(LearningRateSchedule):
  def __init__(self, lr_tensor, lr0, recorder=None):
    """Ctor.

    Args:
      lr_tensor: Tensor variable holding the learning rate.
      lr0: Initial learning rate
      recorder: MeasurementsRecorder that records the overlap.
        If None, set_recorder() should be called before training.
    """
    super(OverlapSchedule, self).__init__(lr_tensor, recorder)
    self.lr0 = lr0
    self.cur_lr = lr0
    self.overlap_upper_threshold = 0.90
    self.overlap_lower_threshold = 0.70
    self.decay_factor = 0.9
    self.dead_time = 20
    self.last_lr_update = 0

  def lr(self):
    if self.step < self.last_lr_update + self.dead_time:
      return self.cur_lr
    if self.recorder.is_recorded(
        meas.HESS_GRAD_OVERLAP, self.step):
      overlap = self.recorder.get_measurement(
        meas.HESS_GRAD_OVERLAP, self.step)
      if overlap > self.overlap_upper_threshold:
        self.cur_lr *= self.decay_factor
        self.last_lr_update = self.step
        tf.logging.info('Overlap crossed threshold, new LR={}'.format(
          self.cur_lr))
      elif overlap < self.overlap_lower_threshold:
        self.cur_lr /= self.decay_factor
        self.last_lr_update = self.step
        tf.logging.info('Overlap crossed threshold, new LR={}'.format(
          self.cur_lr))
    return self.cur_lr
