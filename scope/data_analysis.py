"""Utilities for loading and analyzing experimental results.

A single training run that produces data is called a 'run'. An 'experiment'
consists of one or more runs. Each run is labelled by a set of hyperparameters,
or flags.

During a run, data is stored at a given training
step and can consist of scalars and tensors. Each data point is labeled by a
'tag' string. Tags have a hierarchical, filesystem-like structure.


Synopsis:

# Should be able to read results from local, CNS, Google Storage, BigTable, ...
ds = load_experiments('logs/')

# Show a summary of which experiments we have
ds.show_experiments_summary('name', 'lr', 'batch_size')

# Select a single experiment (will raise if none/more than one match)
single_experiment = ds.select_single(lr=0.1, batch_size=256)

# Select a subset of experiments matching some criteria
interesting_set = ds.select(lr=0.1, optimizer='Adam').select(lambda flags: flags['batch_size'] == 32)

# Print the matching experiments and the available tags
print(interesting_set)

# Get all the events in a big dataframe, including the experiment ID
events = interesting_set.events('loss', 'acc', step_as_index=True)

for experiment in interesting_set:
  events = experiment.events('step', 'loss', 'acc', 'overlap')
  plt.plot(events['step'], events['loss'])
  # For a single experiment, we can access its flags as object members
  plt.title(experiment.name)

# Exploring the tag hierarchy
print(single_experiment.get_tag_tree())

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import pandas

from absl import app
from absl import flags

import scope.tbutils
from scope.experiment_defs import *

FLAGS = flags.FLAGS


def load_experiments(path):
  """Load experimental results.

  Args:
    path:
        Experiment data location, for example a filesystem path.

  Returns:
    An Experiments object.
  """
  return Experiments.from_path(path)


class Experiments:
  """Access to results stored as TensorFlow summaries in a filesystem.

  Assumes the same structure that TensorBoard uses."""

  @staticmethod
  def from_path(path):
    """Create an object that loads experiments from the given path."""
    exp_loader = scope.tbutils.ExperimentLoader(path)
    return Experiments(exp_loader, exp_loader.runs())

  def __init__(self, loader, runs):
    """Ctor.

    Args:
      loader: An ExperimentLoader
      runs: List of selected string run names
    """
    self._loader = loader
    self._runs = runs

  def flags(self):
    """Returns a DataFrame containing the experiment flags.

    DataFrame has one row per run.
    """
    df = pandas.DataFrame()

    for run in self._runs:
      flags = self._loader.flags(run)
      df = df.append(flags, ignore_index=True)

    return df

  def print_summary(self, *args):
    """Print a summary of available experiments.

    Args:
      A list of flags to show.
    """
    flags = self.flags()

    if len(args) == 0:
      print(flags)
    else:
      print(flags[list(args)])

  def __str__(self):
    return str(self.flags())

  def select(self, *args, **kwargs):
    """Select experiments based on criteria.

    Examples:

    ds.select(lr=0.1, optimizer='Adam')
    ds.select(lambda flags: flags['batch_size'] == 32)

    Each argument represents a criterion, and all criteria must match for an
    experiment to be selected.

    Args:
      args: If a function, it is executed on the dictionary of flags and should
        return True if the experiment matches and False otherwise. If a dict,
        should contain  flag: value  pairs that are matched against actual
        values.
      kwargs: flag=value pairs where a given flag must match its given value.

    Returns:
      An ExperimentList of matching experiments.
    """
    matching = []

    for run in self._runs:
      flags = self._loader.flags(run)
      if self._experiment_matches(flags, *args, **kwargs):
        matching.append(run)

    return self.subset(matching)

  def __len__(self):
    return len(self._runs)

  def select_single(self, *args, **kwargs):
    """Select a single experiment based on criteria.

    Same as select(), except it ensures that only a single experiment is chosen.
    Otherwise a ValueError is raised.
    """
    result = self.select(*args, **kwargs)
    if len(result) != 1:
      raise ValueError("Found {} experiments, expected 1".format(len(result)))
    return result

  def subset(self, runs):
    """Returns an Experiments representing a subset of the this object's
    runs.

    Args:
      runs: A list of run names that are a subset of the current runs.

    Returns:
      Experiments.
    """
    return Experiments(self._loader, runs)

  def groupby(self, *args):
    """Group experiments according to the given flags.

    Args:
      args: List of string flags to group by.

    Returns:
      An iterable over Experiments. Each element contains selected experiments
      that share the same values for the flags in args.
    """
    grouped = self.flags().groupby(list(args))
    results = []

    for g in grouped:
      df = g[1]
      runs = df[RUN_NAME_TAG].values
      results.append(self.subset(runs))

    return results

  def events(self, *args, **kwargs):
    """Returns a DataFrame containing all events in the experiment list.

    Args:
      args: List of tags and/or flags to include in the DataFrame. If empty,
        all tags are included but no flags.
      index: If specified, the tag to use as DataFrame index. If None, no
        index is used. By default, 'step' is used as the index.
    """
    if len(args) == 0:
      tags = None
    else:
      tags = args

    df_map = self._loader.events(self._runs, tags)

    df = pandas.DataFrame()
    for run in df_map:
      df = df.append(df_map[run])

    if 'index' in kwargs:
      df.reset_index(inplace=True)
      if kwargs['index'] is not None:
        df.set_index(kwargs['index'], inplace=True)

    return df

  def get_tag_tree(self):
    """Returns a string representation of all the available tags in a
    hierarchical format.
    """
    tags_root = {}

    for run in self._runs:
      tags = self._loader.tags(run)

      for tag in tags:
        tree = tags_root
        for elem in tag.split('/'):
          if not elem in tree:
            tree[elem] = {}
          tree = tree[elem]

    return self._tree_to_str(tags_root)

  def _tree_to_str(self, tree, depth=0):
    """Convert a nested dictionary to a nice string."""
    s = ''
    tab = '    '

    for key in tree:
      s += tab * depth
      s += key
      if len(tree[key]) > 0:
        s += '/'
      s += '\n'
      s += self._tree_to_str(tree[key], depth + 1)

    return s

  def _experiment_matches_func(self, flags, func):
    """Returns whether the given function returns True on the given flags
    dict. If func is not a function, returns None."""
    try:
      try:
        result = func(flags)
      except AttributeError:
        print('Warning: select function raised exception for flags, skipping:')
        print(flags)
        return False
      if type(result) != bool:
        raise ValueError('Expecting boolean return value')
      elif result == False:
        return False
    except TypeError:
      return None

  def _experiment_matches_dict(self, flags, criteria):
    """Returns whether the given flags dict matches the criteria dict.
    If criteria is not a dict, returns None."""
    try:
      for flag, val in criteria.items():
        if flags[flag] != val:
          return False
      return True
    except AttributeError:
      return None

  def _experiment_matches(self, flags, *args, **kwargs):
    for arg in args:
      arg_processed = False

      matches = self._experiment_matches_func(flags, arg)
      if matches is not None:
        arg_processed = True
        if not matches:
          return False

      matches = self._experiment_matches_dict(flags, arg)
      if matches is not None:
        arg_processed = True
        if not matches:
          return False

    return self._experiment_matches_dict(flags, kwargs)

  def __iter__(self):
    self._iter_idx = 0
    return self

  def __next__(self):
    try:
      result = self.subset([self._runs[self._iter_idx]])
      self._iter_idx += 1
      return result
    except IndexError:
      raise StopIteration

  next = __next__  # python 2 compatibility

  def __getattr__(self, name):
    """Access flag values like this: experiments.batch_size

    If there is more than one run, all values of the flag must agree.

    Args:
      name: string flag name

    Raises:
      AttributeError: If flag doesn't exist, or if runs don't all have the
        same flag value.
    """
    flags = self.flags()

    if not name in flags:
      raise AttributeError('Missing attribute (flag) {}'.format(name))

    grouped = flags.groupby(name)

    if len(grouped) != 1:
      raise AttributeError(
          'Multiple values for attribute (flag) {}'.format(name))

    for g in grouped:
      return g[0]


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

if __name__ == '__main__':
  app.run(main)
