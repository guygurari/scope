"""Measure things during training."""

import os
import collections
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import scope.tfutils as tfutils
from scope.tfutils import Timer, NumpyPrintEverything

def _overlap(vec1, vec2):
    """Compute the normalized overlap between two NumPy vectors"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return np.dot(vec1, vec2) / norm1 / norm2


def _save_array(directory, time, name, array):
    np.save('{}/{}_{:05d}.npy'.format(directory, name, time), array)


# Specifies how often to measure. 'freq' is the epoch or step frequency.
# 0 means don't measure, 1 means measure every epoch/step, etc.
# every_step is a boolean, if True then the measurement is performed
# every freq steps, otherwise every freq epochs.
MeasurementFrequency = collections.namedtuple(
    'MeasurementFrequency', ['freq', 'every_step'])


class MeasurementsWriter:
    """Save scalar and tensor summaries."""
    def __init__(self, args, log_dir):
        """args = argument dictionary to log along with logs"""
        super().__init__()
        self.args = args
        self.summary_writer = tf.summary.FileWriter(log_dir)

    def save_summary(self, name, value, step):
        """Save a scalar summary."""
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        self.summary_writer.add_summary(summary, step)

    def save_tensor_summary(self, name, value, step):
        """Save a Tensor summary."""
        summary = tf.Summary()
        tensor_proto = tf.make_tensor_proto(value)
        summary_value = tf.Summary.Value(tag=name, tensor=tensor_proto)
        summary.value.extend([summary_value])
        self.summary_writer.add_summary(summary, step)

    def close(self):
        self.summary_writer.close()


class Measurement(keras.callbacks.Callback):
    """Basic class for performing measurements at intervals
    given by MeasurementFrequency."""
    def __init__(self, freq, writer):
        """freq is MeasurementFrequency"""
        super().__init__()
        self.freq = freq
        self.writer = writer
        self.step = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.batch = 0
        self.epoch = epoch
        if self._should_measure_by_epoch() \
           and not self.freq.every_step:
            self.measure(logs)

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        self.batch = batch
        if self._should_measure_by_step() \
           and self.freq.every_step:
            self.measure(logs)

    def on_batch_end(self, batch, logs=None):
        self.step += 1

    def measure(self, logs=None):
        """To be overidden."""
        pass

    def save_summary(self, name, value):
        """Save a scalar summary at the current step."""
        self.writer.save_summary(name, value, self.step)

    def save_tensor_summary(self, name, value):
        """Save a tensor summary at the current step."""
        self.writer.save_tensor_summary(name, value, self.step)

    @property
    def time_str(self):
        """Returns time string specifying step and epoch."""
        return 'step={} epoch={}'.format(self.step, self.epoch)

    def _should_measure_by_epoch(self):
        return self.freq.freq > 0 \
            and self.epoch % self.freq.freq == 0 \
            and not self.freq.every_step

    def _should_measure_by_step(self):
        return self.freq.freq > 0 \
            and self.step % self.freq.freq == 0 \
            and self.freq.every_step


class BasicMetricsMeasurement(Measurement):
    def __init__(self, writer, model, freq,
                 train_batches, test_batches, show_progress=False):
        super().__init__(freq, writer)
        self.timer = Timer()
        self.train_batches = train_batches
        self.test_batches = test_batches
        self.show_progress = show_progress

        # Find accuracy function, adapted from keras/training.py
        y_true = model.targets[0]
        y_pred = model.output
        output_shape = model.output_shape
        acc_fn = None
        if output_shape[-1] == 1:
            acc_fn = keras.metrics.binary_accuracy
        else:
            acc_fn = keras.metrics.categorical_accuracy

        self.all_tensors = [model.total_loss, K.mean(acc_fn(y_true, y_pred))]
        self.weight_norm = tf.norm([tf.norm(w)
                                    for w in model.trainable_weights])

    def measure(self, logs=None):
        """A Keras callback that collects gradient mean and variance
        statistics."""
        timer = tfutils.Timer()
        logs = logs or {}
        sess = K.get_session()

        self.save_summary('epoch', self.epoch)
        self.save_summary('step', self.step)
        self.save_summary(
            'current_lr', self.model.optimizer.lr.eval(sess))
        self.save_summary(
            'weight_norm', K.get_session().run(self.weight_norm))

        self._compute_metrics(
            self.train_batches, logs, prefix='')
        self._compute_metrics(
            self.test_batches, logs, prefix='val_')

        if self.show_progress:
            self._print_simple_progress(logs)
        tf.logging.info(
            'Timing: Basic metrics: {} secs'.format(timer.secs))

    def _compute_metrics(self, batches, logs, prefix):
        means = tfutils.compute_sample_mean_tensor(
            self.model, batches, self.all_tensors)
        self.save_summary(prefix + 'loss', means[0])
        self.save_summary(prefix + 'acc', means[1])
        logs[prefix + 'loss'] = means[0]
        logs[prefix + 'acc'] = means[1]

    def _print_simple_progress(self, logs):
        duration = '{:.1f} secs'.format(self.timer.secs)

        metric_strs = []

        for metric in ['loss', 'val_loss', 'acc', 'val_acc',
                       'mean_absolute_error',
                       'val_mean_absolute_error']:
            if metric in logs:
                metric_strs.append('{}={:.3f}'.format(
                    metric, logs[metric]))

        tf.logging.info('Epoch {} (step {}): {} ({})'.format(
            self.epoch, self.step,
            ' '.join(metric_strs), duration,
        ))


class GradientMeasurement(Measurement):
    def __init__(self, writer, model, freq,
                 train_batches, test_batches, random_overlap=False):
        """freq is MeasurementFrequency."""
        super().__init__(freq, writer)
        self.model = model
        self.train_batches = train_batches
        self.test_batches = test_batches
        self.random_overlap = random_overlap
        self._create_gradient_tensors(model)

    def _create_gradient_tensors(self, model):
        tf.logging.info('Creating gradient tensors...')

        self.weights = model.trainable_weights
        self.all_tensors = {}

        # Prepare some tensors. Here we create tensors that hold
        # all elements of vectors such as the gradient.
        # This allows us to compute mean and variance.

        # Holds a list, each element is the gradient of a layer
        grad = tf.gradients(model.total_loss, self.weights)
        flat_grad = tfutils.flatten_tensor_list(grad)
        self.all_tensors['gradient'] = flat_grad

        # Hessian-gradient product
        self.v = tf.placeholder(
            tf.float32, shape=(tfutils.total_num_weights(model),))
        self.Hv = tfutils.hessian_vector_product(
            model.total_loss, model.trainable_weights, self.v)

        # grad_norm_sqr = tf.reduce_sum(
        #     [tf.reduce_sum(g * g) for g in grad])
        # s_hess_grad = 0.5 * tfutils.flatten_tensor_list(
        #     tf.gradients(grad_norm_sqr, self.weights))
        # self.all_tensors['hessian_gradient'] = s_hess_grad

    def measure(self, logs=None):
        """A Keras callback that collects gradient mean and variance
        statistics."""
        logs = logs or {}
        tf.logging.info(
            '\nComputing gradients at epoch {} (batch {})...'.format(
                self.epoch, self.batch))
        timer = tfutils.Timer()
        tf.logging.info('Training gradients ...')
        train_stats, self.full_batch_g, self.full_batch_Hg = (
            self._compute_gradients(
                self.train_batches, logs, prefix='', prnt=True))

        if self.random_overlap:
            self._compute_Hrand(self.train_batches, logs,
                                prefix='', prnt=True)

        tf.logging.info('Test gradients ...')
        self._compute_gradients(
            self.test_batches, logs, prefix='val_', prnt=False)
        tf.logging.info('Timing: Gradients: {} secs'.format(timer.secs))

    def _compute_gradients(self, batches, logs, prefix, prnt):
        # timer = tfutils.Timer()
        stats = {name: tfutils.TensorStatistics(t)
                 for (name, t) in self.all_tensors.items()}
        full_batch_g = None

        batch_idx = 0
        while True:
            # tf.logging.info('batch_idx =', batch_idx)
            x_batch, y_batch = batches.next_batch()
            results = tfutils.keras_compute_tensors(
                self.model, x_batch, y_batch, self.all_tensors)

            for name, value in results.items():
                stats[name].add_minibatch(value)

            g_sum = results['gradient'] * len(x_batch)
            if full_batch_g is None:
                full_batch_g = np.array(g_sum)
            else:
                full_batch_g += g_sum

            batch_idx += 1
            if batches.at_start_of_epoch():
                break
        assert batch_idx == batches.batches_per_epoch
        full_batch_g /= batches.N

        # tf.logging.info('Gradients took {} secs for {} batches, '
        #                 '{} sec/sample'.format(
        #                     timer.secs,
        #                     batches.batches_per_epoch,
        #                     timer.secs / batches.batches_per_epoch))

        self._save_statistics(stats, logs, prefix, prnt)
        full_batch_Hg = self._compute_Hg(
            batches, full_batch_g, logs, prefix, prnt)
        return stats, full_batch_g, full_batch_Hg

    def _compute_Hv_overlap(self, batches, v):
        Hv = tfutils.compute_sample_mean_tensor(
            self.model, batches, self.Hv, {self.v: v})

        v_norm = np.linalg.norm(v)
        Hv_norm = np.linalg.norm(Hv)

        Hv_dot_v = np.dot(Hv, v)
        Hv_v_overlap = Hv_dot_v / v_norm / Hv_norm
        Hv_eigenvalue = Hv_dot_v / v_norm**2
        return Hv, Hv_v_overlap, Hv_eigenvalue

    def _compute_Hg(self, batches, full_batch_g, logs, prefix, prnt):
        Hg, Hg_g_overlap, Hg_eigenvalue = self._compute_Hv_overlap(
            batches, full_batch_g)

        if prnt:
            tf.logging.info(
                'Hg_eigenvalue ={}\tHg_g_overlap ={}'.format(
                    Hg_eigenvalue, Hg_g_overlap))
        self.save_summary(
            prefix + 'hessian_gradient/gradient_overlap', Hg_g_overlap)
        self.save_summary(
            prefix + 'hessian_gradient/eigenvalue', Hg_eigenvalue)
        return Hg

    def _compute_Hrand(self, batches, logs, prefix, prnt):
        v = np.random.randn(tfutils.total_num_weights(self.model)).astype(
            np.float32)

        Hv, Hv_v_overlap, Hv_eigenvalue = self._compute_Hv_overlap(batches, v)

        if prnt:
            tf.logging.info(
                'Hv_eigenvalue ={}\tHv_v_overlap ={}'.format(
                    Hv_eigenvalue, Hv_v_overlap))
        self.save_summary(
            prefix + 'hessian_gradient/random_overlap', Hv_v_overlap)
        self.save_summary(
            prefix + 'hessian_gradient/random_eigenvalue', Hv_eigenvalue)

    def _save_statistics(self, stats, logs, prefix='', prnt=True):
        for name, stat in stats.items():
            full_name = prefix + name
            self.save_summary(
                full_name + '/mean_norm', stat.norm_of_mean)
            self.save_summary(
                full_name + '/std_norm', stat.norm_of_std)
            self.save_summary(
                full_name + '/snr',
                stat.norm_of_mean / stat.norm_of_std)


class LanczosHessianMeasurement(Measurement):
    """Measure part of the Hessian spectrum."""
    def __init__(self, writer, model, freq, num_evs,
                 x_train, y_train, batch_size,
                 lr, log_dir, grad_measurement=None):
        """grad_measurement is GradientMeasurement"""
        super().__init__(freq, writer)
        self.model = model
        self.num_evs = num_evs
        self.lr = lr
        self.grad_measurement = grad_measurement
        self.hessian_spec = tfutils.KerasHessianSpectrum(
            model, x_train, y_train, batch_size)
        self.prev_evecs = None

        self.detailed_log_dir = os.path.join(log_dir, 'lanczos_hessian')
        os.makedirs(self.detailed_log_dir)

    def measure(self, logs):
        """Compute parts of the Hessian spectrum"""
        tf.logging.info(
            'Computing {} H evs with Lanczos ...'.format(self.num_evs))
        evals, evecs = self.hessian_spec.compute_spectrum(
            self.num_evs, show_progress=True)

        secs_per_iter = self.hessian_spec.lanczos_secs \
                        / self.hessian_spec.lanczos_iterations
        tf.logging.info('Hessian took {:.2f} secs, {} Lanczos iterations, '
                        '({:.2f} secs/iteration)'.format(
                            self.hessian_spec.lanczos_secs,
                            self.hessian_spec.lanczos_iterations,
                            secs_per_iter,
                        ))

        _save_array(self.detailed_log_dir,
                    self.step, 'H_evals', evals)
        _save_array(self.detailed_log_dir,
                    self.step, 'H_evecs', evecs)

        # self.save_summary('H_evs', evals)

        if self.grad_measurement is not None:
            g = self.grad_measurement.full_batch_g
            unit_g = g / np.linalg.norm(g)
            # Take the absolute value because evec has arbitrary
            # orientation.
            overlaps = np.abs(evecs.transpose().dot(unit_g))
            explained = 0

            tf.logging.info('\teval\tc_i\tc_i^2\t%explained')
            tf.logging.info('---------------------------------------')
            zipped = zip(range(len(evals)),
                         reversed(evals),
                         reversed(overlaps),
                         reversed(overlaps**2))
            for i, ev, ov, ov_sqr in zipped:
                explained += ov_sqr
                tf.logging.info('{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.1f}'.format(
                    i + 1, ev, ov, ov_sqr, 100 * explained))
            tf.logging.info('---------------------------------------')

            # self.save_summary('Hvec_g_overlaps', overlaps)
            _save_array(self.detailed_log_dir, self.step, 'g', g)
            _save_array(self.detailed_log_dir, self.step, 'Hg',
                        self.grad_measurement.full_batch_Hg)
            _save_array(
                self.detailed_log_dir, self.step,
                'Hvec_g_overlaps', overlaps)

        if self.prev_evecs is not None:
            VVprime = evecs.transpose().dot(self.prev_evecs)
            _save_array(self.detailed_log_dir, self.step,
                        'Hvec_VVp_overlaps', VVprime)
            # self.save_summary('Hvec_self_overlaps', np.diag(VVprime))
            tf.logging.info('V^T . V_prev:')

            prev_options = np.get_printoptions()
            np.set_printoptions(formatter={'float': '{:0.2f}'.format})
            tf.logging.info(np.diag(VVprime))
            np.set_printoptions(prev_options)

        self.prev_evecs = evecs


class FullHessianMeasurement(Measurement):
    """Measure part of the Hessian spectrum."""
    def __init__(self, writer, model, freq, train_batches, log_dir,
                 num_eigenvector_correlations,
                 grad_measurement=None):
        """
        :param freq: MeasurementFrequency
        :param num_eigenvector_correlations: Number of leading eigenvectors
        to include when computing correlations between subsequent eigenvectors.
        0 for none.
        :param grad_measurement: A GradientMeasurement object
        """
        super().__init__(freq, writer)
        self.model = model
        self.batches = train_batches
        self.log_dir = log_dir
        self.num_eigenvector_correlations = num_eigenvector_correlations
        self.grad_measurement = grad_measurement
        self.prev_top_V = None
        if log_dir is None:
            self.detailed_log_dir = None
        else:
            self.detailed_log_dir = os.path.join(log_dir, 'full_hessian')
            os.makedirs(self.detailed_log_dir)
        tf.logging.info('Creating full Hessian tensors ...')
        self.hessian_blocks = tfutils.hessian_tensor_blocks(
            model.total_loss, model.trainable_weights)

    def compute_hessian(self):
        hess = None
        batch_idx = 0
        while True:
            tf.logging.info('batch_idx = {}'.format(batch_idx))
            x_batch, y_batch = self.batches.next_batch()
            hess_batch_blocks = tfutils.keras_compute_tensors(
                self.model, x_batch, y_batch, self.hessian_blocks)
            tf.logging.info('hessian_combine_blocks')
            hess_batch = tfutils.hessian_combine_blocks(hess_batch_blocks)

            # Undo mini-batch mean
            hess_batch *= len(x_batch)

            if hess is None:
                hess = hess_batch
            else:
                hess += hess_batch

            batch_idx += 1
            if self.batches.at_start_of_epoch():
                break

        # Do full-batch mean
        hess /= self.batches.N
        return hess

    def measure(self, logs=None):
        """Compute the full Hessian spectrum"""
        tf.logging.info('Computing full Hessian ...')
        timer = Timer()
        hess = self.compute_hessian()
        tf.logging.info(
            'Full Hessian {} took {} secs'.format(hess.shape, timer.secs))
        tf.logging.info('Diagonalizing ...')
        timer = Timer()
        # V columns are eigenvectors
        D, V = np.linalg.eigh(hess)
        tf.logging.info('Diagonalizing took {} secs'.format(timer.secs))

        self._save_array('H_mat', hess)
        self._save_array('H_eigenvectors', V)

        tf.logging.info('Found {} eigenvalues'.format(len(D)))
        self._save_array('H_evs', D)
        self.save_tensor_summary('full_hessian/eigenvalues', D)

        if self.num_eigenvector_correlations > 0:
            top_V = V[:, -self.num_eigenvector_correlations:]
            if self.prev_top_V is not None:
                # Correlations are measured between -1, 1.
                # Detailed correlations are for v_i(t)^T v_i(t+1)
                # Summary correlations are Tr( V(t)^T V(t+1) )
                corr = np.matmul(top_V.transpose(), self.prev_top_V)
                tf.logging.info('H_top_evecs correlations:\n{}'.format(corr))
                tf.logging.info('Diagonal part: {}'.format(corr.diagonal()))
                self._save_array(
                    'H_top_evec_corr', corr)
                # self.save_summary(
                #     'full_hessian/top_evec_correlations', corr)
            self.prev_top_V = top_V

        if self.grad_measurement is not None:
            mean_grad = self.grad_measurement.full_batch_g
            g = mean_grad / np.linalg.norm(mean_grad)
            self._save_array('g', g)

            overlaps = V.transpose().dot(g)
            with NumpyPrintEverything():
                tf.logging.info(
                    'H_top_evec_g_overlaps = {}'.format(
                        overlaps[-self.num_eigenvector_correlations:]))
            self._save_array('H_g_overlaps', overlaps)
            # self.save_summary('full_hessian/g_overlaps', overlaps)

    def _save_array(self, name, arr):
        if self.detailed_log_dir is not None:
            _save_array(self.detailed_log_dir, self.step, name, arr)


class GaussiansMeasurement(Measurement):
    """Measure basic quantities when using Gaussian mixture."""
    def __init__(self, writer, model, freq,
                 x_train, y_train,
                 grad_measurement=None):
        """grad_measurement is GradientMeasurement"""
        super().__init__(freq, writer)
        self.model = model
        self.x_train = x_train
        self.grad_measurement = grad_measurement

    def measure(self, logs):
        """Measure Gaissian overlaps."""
        tf.logging.info('Computing Gaussian stuff')
        params = K.get_session().run(self.model.trainable_weights)
        weights = params[0]
        x1 = self.x_train[0, :]
        x2 = self.x_train[-1, :]
        theta1 = weights[:, 0]
        theta2 = weights[:, 1]

        # biases = params[1]
        # b1 = biases[0]
        # b2 = biases[1]
        b1 = 0
        b2 = 0

        def overlap(v1, v2):
            return v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2)

        ov1 = overlap(theta1, x1 - x2)
        ov2 = overlap(theta2, x1 - x2)

        factor1 = np.exp((theta1 - theta2).dot(x2) + b1 - b2)
        factor2 = np.exp((theta2 - theta1).dot(x1) + b2 - b1)
        tf.logging.info('(x1,x2)={:.3f} ov1={:.3f} ov2={:.3f} '
                        '|theta1|={:.3f} |theta2|={:.3f} '
                        'factor1={:3f} factor2={:3f}'.format(
                            overlap(x1, x2), ov1, ov2,
                            np.linalg.norm(theta1), np.linalg.norm(theta2),
                            factor1, factor2))
        # tf.logging.info('biases=', biases)
        # tf.logging.info('exp(biases)=', np.exp(biases))

        # if self.grad_measurement is not None:
        #     g = self.grad_measurement.full_batch_g


class WeightNormMeasurement(Measurement):
    """Measure norm of weights by layer."""
    def __init__(self, writer, model, freq):
        """
        :param freq: MeasurementFrequency
        """
        super().__init__(freq, writer)
        self.model = model
        self.weight_norms = {w.name: tf.norm(w) for w in model.weights}

    def measure(self, logs=None):
        """Measure weight norm by layer."""
        timer = tfutils.Timer()
        norms = K.get_session().run(self.weight_norms)
        for name, norm in norms.items():
            self.save_summary('weight_norm/' + name, norm)
        tf.logging.info(
            'Timing: Weight norm: {} secs'.format(timer.secs))


class ProjectedGradientDescent(keras.optimizers.SGD):
    """Gradient descent optimizer that projects that gradient
    onto the top Hessian subspace.
       lr: float >= 0. Learning rate.
    """
    def __init__(self, lr, model, x_train, y_train,
                 hessian_spectrum, subspace_dim, **kwargs):
        """hessian_spectrum is a tfutils.KerasHessianSpectrum"""
        super(ProjectedGradientDescent, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')

        self.model = model
        self.hessian_spec = hessian_spectrum
        self.subspace_dim = subspace_dim
        self.x_train = x_train
        self.y_train = y_train

        # self.projector = tf.Variable()

        self.proj_grads = [
            tf.placeholder(tf.float32, shape=w.shape)
            for w in model.trainable_weights]

    # @keras.legacy.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)

        flat_grad = tfutils.flatten_tensor_list(grads)
        flat_grad_eval = tfutils.keras_compute_tensors(
            self.model, self.x_train, self.y_train, flat_grad)

        # Project
        evals, evecs = self.hessian_spec.compute_spectrum(
            self.subspace_dim, show_progress=True)

        flat_grads_projected = np.matmul(evecs,
                                         np.matmul(np.transpose(evecs),
                                                   flat_grad_eval))

        # Reshape from flat back to original shape
        grads_projected = tfutils.unflatten_tensor_list(
            flat_grads_projected, grads)

        self.updates = [K.update_add(self.iterations, 1)]

        for p, g in zip(params, grads_projected):
            new_p = p - self.lr * g
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr))}
        base_config = super(ProjectedGradientDescent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
