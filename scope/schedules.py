"""Implement learning rate schedules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
