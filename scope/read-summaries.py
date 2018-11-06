#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

for summary in tf.train.summary_iterator(sys.argv[1]):
    print(summary)
