#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scope.tbutils import EventLoader

logdir = 'logs/'

el = EventLoader(logdir)

print('runs:', el.runs_map)

events = el.events(el.runs,
                   ['epoch', 'loss', 'acc', 'full_hessian/eigenvalues'])

for run, data in events.items():
  print(run)
  print(data)
  print('')
