#!/usr/bin/env python3

"""Rename all experiments that match a given name.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import glob

from absl import app
from absl import flags

import scope.tbutils

FLAGS = flags.FLAGS

flags.DEFINE_string('logdir', 'logs', 'Base logs directory')
flags.DEFINE_string('from_name', None, 'Original experiment name')
flags.DEFINE_string('to_name', None, 'New experiment name')

paths_to_rename = []

def main(_):
    for path in glob.glob(FLAGS.logdir + '/*'):
        if os.path.isdir(path):
            run_flags = scope.tbutils.load_run_flags(path)
            if run_flags['name'] == FLAGS.from_name:
                print('Renaming path', path)
                run_flags['name'] = FLAGS.to_name
                scope.tbutils.save_run_flags(path, run_flags)
                paths_to_rename.append(path)

    for path in paths_to_rename:
        new_path = re.sub(r'\b{}\b'.format(FLAGS.from_name),
                          FLAGS.to_name,
                          path)
        os.rename(path, new_path)

if __name__ == '__main__':
    flags.mark_flag_as_required('from_name')
    flags.mark_flag_as_required('to_name')
    app.run(main)
