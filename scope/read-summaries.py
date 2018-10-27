#!/usr/bin/env python3

import sys
import tensorflow as tf

for summary in tf.train.summary_iterator(sys.argv[1]):
    print(summary)
