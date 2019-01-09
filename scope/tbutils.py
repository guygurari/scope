"""TensorBoard utilities, for processing saved TensorFlow events.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorboard.backend.event_processing.event_multiplexer as emux
import tensorboard.backend.event_processing.event_accumulator as event_accumulator
import tensorflow as tf
import pandas as pd
import json

from absl import flags

from scope.experiment_defs import *

FLAGS_FILE = 'flags.json'
FLAGS = flags.FLAGS


def _get_flags_dict():
  """Returns a dictionary of flag values."""
  flags_dict = {}
  flags_dict_by_module = FLAGS.flags_by_module_dict()
  for module, val in flags_dict_by_module.items():
    module_flags_dict = {flag.name: flag.value for flag in val}
    flags_dict.update(module_flags_dict)
  return flags_dict


def save_run_flags(run_logdir, flags_dict=None, additional_flags=None):
  """Save the experiment's flags.

    Args:
        rundir: The run's log dir.
        flags_dict: Dict containing flag values. If None, uses the program
          flags.
        additional_flags: Dict containing additional flags. Flags appearing here
          will overwrite those in flags_dict.

    Returns:
        A dict containing the experiment's flags.
    """
  if flags_dict is None:
    flags_dict = _get_flags_dict()
  else:
    flags_dict = dict(flags_dict)
  if additional_flags is not None:
    flags_dict.update(additional_flags)
  flags_as_json = json.dumps(flags_dict)
  with tf.gfile.GFile('{}/{}'.format(run_logdir, FLAGS_FILE), 'w') as f:
    f.write(flags_as_json)


def load_run_flags(run_logdir):
  """Load the experiment's flags.

    Args:
        rundir: The run's log dir.
        full_flags: If False (default), only load the main module's flags.
        If True, load all flags set by all modules.

    Returns:
        A dict containing the experiment's flags.
    """
  with tf.gfile.GFile('{}/{}'.format(run_logdir, FLAGS_FILE), 'r') as f:
    return json.load(f)


class EventLoader:
  """Reads scalar and tensor events written in TensorBoard format and
    converts them to DataFrames.
  """

  def __init__(self, logdir):
    """Loads the events of all runs under the given directory.

        The directory should contain runs and event files in the same format
        used by TensorBoard.

        Args:
          logdir: The string directory name containing event files.
    """
    self.logdir = logdir

    # Do not restrict the number of scalars and tensors we're saving
    size_guidance = event_accumulator.DEFAULT_SIZE_GUIDANCE
    size_guidance[event_accumulator.SCALARS] = 0
    size_guidance[event_accumulator.TENSORS] = 0

    self.event_mux = emux.EventMultiplexer(size_guidance=size_guidance)
    self.event_mux.AddRunsFromDirectory(self.logdir)
    self.event_mux.Reload()

  def reload(self):
    """Reloads the new events that have been added since the object

        was constructed or since the last reload() was called.
    """
    self.event_mux.Reload()

  def tags(self, run):
    """Returns a list of string tag names for the given run.

        Args:
            run: string run name.
    """
    runs = self.event_mux.Runs()
    run_tags = runs[run]
    return run_tags['scalars'] + run_tags['tensors']

  def runs(self):
    """Returns a list of available string run names."""
    return list(self.event_mux.Runs().keys())

  def run_dirs(self):
    """Returns a dict mapping run names to run directories."""
    return self.event_mux.RunPaths()

  def _single_run_events(self, run, tags):
    """Returns an events DataFrame for a single run."""
    # Start with a DataFrame that includes all the steps and their wall
    # times, by looking at the STEP_TAG tag. We record this tag at every
    # step where every other measurement takes place.
    #
    # This makes joining with other measurements easier, because all
    # measurements include a wall time, and we want to have a reference
    # wall time.
    try:
      step_scalars = self.event_mux.Scalars(run, STEP_TAG)
    except KeyError:
      raise ValueError('Run {} does not seem to contain any events'.format(run))
    run_df = pd.DataFrame([[s.step, s.wall_time] for s in step_scalars],
                          columns=[STEP_TAG, WALL_TIME_TAG])
    run_df.set_index(STEP_TAG, inplace=True)

    if tags is None:
      run_tags = self.tags(run)
    else:
      run_tags = list(tags)

    # Do not include 'step' tag because it's already the index,
    # and it messes up the DataFrame.
    if STEP_TAG in run_tags:
      run_tags.remove(STEP_TAG)

    for tag in run_tags:
      try:
        scalars = self.event_mux.Scalars(run, tag)
        scalar_data = [[s.step, s.value] for s in scalars]
        run_df = self._join_with_timeseries(run_df, scalar_data, tag)
        scalar_found = True
      except KeyError:
        scalar_found = False

      try:
        tensors = self.event_mux.Tensors(run, tag)
        tensor_data = [
            [t.step, tf.make_ndarray(t.tensor_proto)] for t in tensors
        ]
        run_df = self._join_with_timeseries(run_df, tensor_data, tag)
        tensor_found = True
      except KeyError:
        tensor_found = False

      if not scalar_found and not tensor_found:
        raise ValueError('Tag {} not found as scalar or tensor'.format(tag))

    return run_df

  def events(self, runs=None, tags=None):
    """Returns a map of event data, mapping each run name to a DataFrame

        containing the run's events.

        Args:
          runs: A sequence of string run names. If None, use all runs.
          tags: A sequence of string tag names. If None, all tags are included.

        Returns:
          A map run -> DataFrame, where the DataFrame contains the relevant
          events with the given tags as columns. Events of scalar and tensor
          types are included.
        """
    if runs is None:
      runs = self.runs()
    result = {}
    for run in runs:
      result[run] = self._single_run_events(run, tags)
    return result

  def _join_with_timeseries(self, df, time_series, tag):
    ts_df = pd.DataFrame(time_series, columns=[STEP_TAG, tag])
    ts_df.set_index(STEP_TAG, inplace=True)
    return df.join(ts_df, how='outer')


class ExperimentLoader:
  """Loads experiment events and flags."""

  def __init__(self, logdir):
    """Loads all events under the given directory.

        The directory should contain runs and event files in the same format
        used by TensorBoard, plus the flags used to run each experiment
        as saved by `save_run_flags`.

        Args:
          logdir: The string directory name containing event files.
    """
    self.logdir = logdir
    self.event_loader = EventLoader(logdir)
    self._load_flags()

  def reload(self):
    """Reload the experiment data without reloading existing events."""
    self.event_loader.reload()
    self._load_flags()

  def runs(self):
    """Returns a list of loaded string run names."""
    return self.event_loader.runs()

  def tags(self, run):
    """Returns a list of string tag names for the given run.

        Args:
            run: string run name.
    """
    return self.event_loader.tags(run)

  def flags(self, run):
    """Returns the flag dictionary for the given run."""
    return self._flags[run]

  def events(self, runs=None, tags=None):
    """Returns a map of event data, mapping each run name to a DataFrame

        containing the run's events.

        Args:
          runs: A sequence of string run names. If None, include all runs. If
            it's a dict, it is treated as criteria for `select`ing the
            experiments.
          tags: A sequence of string tag names. If None, include all tags.

        Returns:
          A map run -> DataFrame, where the DataFrame contains the relevant
          events with the given tags as columns. Events of scalar and tensor
          types are included.
        """
    if type(runs) is dict:
      runs = self.select(criteria=runs)
    return self.event_loader.events(runs, tags)

  def single_run_events(self, run, tags=None):
    """Returns the DataFrame of events for the given run."""
    return self.events(runs=[run], tags=tags)[run]

  def select(self, criteria):
    """Select experiments based on the given flag criteria.

        Args:
            criteria: A dict mapping flag names to desired values. All flag
              values must match.

        Returns:
            A list of run names matching the given criteria.
        """
    result = []
    for run in self.runs():
      if self._flags_match_criteria(self.flags(run), criteria):
        result.append(run)
    return result

  def _flags_match_criteria(self, run_flags, criteria):
    """Returns whether the given flags dict matches the criteria dict."""
    for flag, val in criteria.items():
      if run_flags[flag] != val:
        return False
    return True

  def _load_flags(self):
    """Load the flag files of all experiments."""
    self._flags = {}
    for run, path in self.event_loader.run_dirs().items():
      self._flags[run] = load_run_flags(path)
      self._flags[run][RUN_NAME_TAG] = run
