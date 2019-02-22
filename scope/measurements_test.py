#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import scipy
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K

import scope.measurements as measurements

import colored_traceback
colored_traceback.add_hook()

precision = 5


class TestMeasurements(unittest.TestCase):
  def test_frequency_from_string(self):
    f = measurements.Frequency.from_string('10')
    self.assertEqual(f.freq, 10)
    self.assertEqual(f.stepwise, False)

    f = measurements.Frequency.from_string('11e')
    self.assertEqual(f.freq, 11)
    self.assertEqual(f.stepwise, False)

    f = measurements.Frequency.from_string('12epoch')
    self.assertEqual(f.freq, 12)
    self.assertEqual(f.stepwise, False)

    f = measurements.Frequency.from_string('13epochs')
    self.assertEqual(f.freq, 13)
    self.assertEqual(f.stepwise, False)

    f = measurements.Frequency.from_string('14s')
    self.assertEqual(f.freq, 14)
    self.assertEqual(f.stepwise, True)

    f = measurements.Frequency.from_string('15step')
    self.assertEqual(f.freq, 15)
    self.assertEqual(f.stepwise, True)

    f = measurements.Frequency.from_string('16steps')
    self.assertEqual(f.freq, 16)
    self.assertEqual(f.stepwise, True)

    with self.assertRaises(ValueError):
      measurements.Frequency.from_string('bad')


def main(_):
  unittest.main()


if __name__ == '__main__':
  tf.app.run(main)
