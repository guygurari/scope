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

import scope.tfutils as tfutils
import scope.lanczos as lanczos
import scope.measurements as measurements

import colored_traceback
colored_traceback.add_hook()

precision = 5


class TestTensorFlowUtils(unittest.TestCase):

  def test_unflatten_tensor_list(self):
    tensors = []
    tensors.append(tf.constant([[1, 2, 3], [4, 5, 6]]))
    tensors.append(tf.constant([[-1], [-2]]))
    tensors.append(tf.constant(12))

    flat = tfutils.flatten_tensor_list(tensors)
    unflat = tfutils.unflatten_tensor_list(flat, tensors)

    self.assertTrue(len(flat.shape.dims) == 1)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      tensors_eval = sess.run(tensors)
      unflat_eval = sess.run(unflat)
      self.assertEqual(len(tensors_eval), len(unflat_eval))
      for t, u in zip(tensors_eval, unflat_eval):
        self.assertTrue(np.array_equal(t, u))

  def test_jacobian(self):
    m_np = np.array([[1., 2.], [3., 4.]], np.float32)
    m = tf.Variable(m_np)
    x = tf.Variable([4., -1.], tf.float32)
    y = tf.einsum('nm,m->n', m, x)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      jacobian = tfutils.jacobian(y, x)
      jacobian_actual = sess.run(jacobian)
      self.assertTrue(np.allclose(jacobian_actual, m_np))

  def test_jacobian_dynamic_dim(self):
    m_np = np.array([[1., 2.], [3., 4.]], np.float32)

    # m = tf.Variable(m_np)
    # x = tf.Variable([4., -1.], tf.float32)

    m = tf.placeholder(tf.float32, shape=[2, None])
    x = tf.placeholder(tf.float32, shape=[None])

    y = tf.einsum('nm,m->n', m, x)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      jacobian = tfutils.jacobian(y, x)
      jacobian_actual = sess.run(jacobian, feed_dict={m: m_np, x: [4., -1.]})
      self.assertTrue(np.allclose(jacobian_actual, m_np))

  def test_jacobian_multirank_y(self):
    m_np = np.array([[1., 2.], [3., 4.]], np.float32)
    m = tf.Variable(m_np)
    x = tf.Variable([4.], tf.float32)
    y = m * x

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      jacobian = tfutils.jacobian(y, x)
      jacobian_actual = sess.run(jacobian)
      jacobian_actual = np.reshape(jacobian_actual, m_np.shape)
      self.assertTrue(np.allclose(jacobian_actual, m_np))

  def test_trace_hessian_reference(self):
    a_val = 1.
    b_val = 2.
    a = tf.Variable(a_val)
    b = tf.Variable(b_val)
    f = 6 * a * a * b + a * a + 7 * a * b + 3 * b * b + 13 * a + 15 * b + 5
    trH = 2 * 6 * b_val + 2 * (1 + 3)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      trH_actual = sess.run(tfutils.trace_hessian_reference(f, [a, b]))
      self.assertAlmostEqual(trH, trH_actual, precision)

  # TODO fix this test
  # def test_trace_hessian_nn_softmax_crossentropy(self):
  #     num_samples = 1
  #     input_dim = 3
  #     num_classes = 10

  #     x_train = np.random.rand(num_samples, input_dim)
  #     y_train = np.random.rand(num_samples, num_classes)

  #     X = tf.placeholder(tf.float32, [num_samples, input_dim])
  #     Y = tf.placeholder(tf.float32, [num_samples, num_classes])

  #     feed = {X: x_train, Y: y_train}

  #     h1 = tf.layers.dense(X, 4, activation=tf.nn.relu, name='h1')
  #     h2 = tf.layers.dense(h1, 6, activation=tf.nn.relu, name='h2')
  #     logits = tf.layers.dense(
  #         h2, num_classes, activation=None, name='logits')

  #     # loss = tf.losses.mean_squared_error(Y, logits)
  #     loss = tf.nn.softmax_cross_entropy_with_logits(
  #         labels=Y, logits=logits)

  #     w1 = tf.get_default_graph().get_tensor_by_name(
  #         os.path.split(h1.name)[0] + '/kernel:0')
  #     b1 = tf.get_default_graph().get_tensor_by_name(
  #         os.path.split(h1.name)[0] + '/bias:0')
  #     w2 = tf.get_default_graph().get_tensor_by_name(
  #         os.path.split(h2.name)[0] + '/kernel:0')
  #     b2 = tf.get_default_graph().get_tensor_by_name(
  #         os.path.split(h2.name)[0] + '/bias:0')
  #     weights = [w1, b1, w2, b2]

  #     with tf.Session() as sess:
  #         sess.run(tf.global_variables_initializer())

  #         # J = tfutils.jacobian(tf.reshape(logits, [-1]), w1)
  #         # H = tfutils.hessians(loss, logits)
  #         # print('H[0].shape =', H[0].shape)

  #         # Expected Tr(H)
  #         trH = sess.run(
  #             tfutils.trace_hessian_reference(loss, weights), feed)

  #         # Actual Tr(H)
  #         trH_t = tfutils.trace_hessian(loss, logits, weights)
  #         trH_actual = sess.run(trH_t, feed)
  #         self.assertAlmostEqual(trH, trH_actual, precision)

  #         trH_sce_t = tfutils.trace_hessian_softmax_crossentropy(
  #             logits, weights)
  #         trH_sce_actual = sess.run(trH_sce_t, feed)
  #         self.assertAlmostEqual(trH, trH_sce_actual, precision)

  def test_gradients_and_hessians(self):
    tf.reset_default_graph()

    n = 50
    d = 5

    x = np.random.random((n, d)).astype(np.float32)
    y = np.random.random(n).astype(np.float32)

    w = np.random.random(d).astype(np.float32)

    # NumPy
    def f(x_sample):
      return x_sample.dot(w)

    loss = np.linalg.norm(x.dot(w) - y)**2 / (2 * n)
    grad = np.zeros(d)
    term1 = np.zeros(d)

    for i in range(d):
      for a in range(n):
        grad[i] += (f(x[a, :]) - y[a]) * x[a, i] / n

    grad_norm_sqr = np.linalg.norm(grad)**2

    for j in range(d):
      for a in range(n):
        for b in range(n):
          for i in range(d):
            # x_a = x[a, :]
            x_b = x[b, :]
            term1[j] += (f(x_b) - y[b]) \
                        * x[a, i] * x[b, i] * x[a, j]
    term1 *= 2 / n**2

    # hessians = np.sum(x * x, axis=0) / n
    tr_hessian = np.sum(x * x) / n

    # TensorFlow
    with tf.Session() as sess:
      x_t = tf.placeholder(tf.float32, shape=x.shape)
      y_t = tf.placeholder(tf.float32, shape=y.shape)

      w_t = tf.Variable(w, name='w')

      sess.run(tf.global_variables_initializer())

      logits_t = tf.einsum('ai,i->a', x_t, w_t)
      loss_t = tf.norm(logits_t - y_t)**2 / (2 * n)

      grad_t = tf.gradients(loss_t, w_t)
      grad_norm_sqr_t = tf.reduce_sum([tf.reduce_sum(g * g) for g in grad_t])
      term1_t = tf.gradients(grad_norm_sqr_t, w_t)

      # This gives the whole Hessian matrix
      hessians_t = tf.hessians(loss_t, w_t)
      hessians_t = hessians_t[0]
      tr_hessian_t = tf.trace(hessians_t)

      #############################################################
      # Compute trace of Hessian in a way that scales to non-vector
      # weights
      tr_hessian_2_t = tfutils.trace_hessian(loss_t, logits_t, w_t)

      ##############################################################

      results = sess.run((loss_t, grad_t, grad_norm_sqr_t, term1_t,
                          tr_hessian_t, tr_hessian_2_t),
                         feed_dict={
                             x_t: x,
                             y_t: y
                         })

      loss_result, grad_result, grad_norm_sqr_result, \
          term1_result, tr_hessian_result, \
          tr_hessian_2_result = results

      self.assertAlmostEqual(loss, loss_result, precision)
      self.assertAlmostEqual(grad_norm_sqr, grad_norm_sqr_result, precision)

      for term1_elem, term1_result_elem in zip(term1, term1_result[0]):
        self.assertAlmostEqual(term1_elem, term1_result_elem, precision)

      self.assertAlmostEqual(tr_hessian, tr_hessian_result, precision)
      self.assertAlmostEqual(tr_hessian, tr_hessian_2_result, precision)

      for g_elem, g_result_elem in zip(grad, grad_result[0]):
        self.assertAlmostEqual(g_elem, g_result_elem, precision)

  def test_gradients_and_hessians_dynamic_dim(self):
    tf.reset_default_graph()

    n = 50
    d = 5

    x = np.random.random((n, d)).astype(np.float32)
    y = np.random.random(n).astype(np.float32)

    w = np.random.random(d).astype(np.float32)

    # NumPy
    def f(x_sample):
      return x_sample.dot(w)

    loss = np.linalg.norm(x.dot(w) - y)**2 / (2 * n)
    grad = np.zeros(d)
    term1 = np.zeros(d)

    for i in range(d):
      for a in range(n):
        grad[i] += (f(x[a, :]) - y[a]) * x[a, i] / n

    grad_norm_sqr = np.linalg.norm(grad)**2

    for j in range(d):
      for a in range(n):
        for b in range(n):
          for i in range(d):
            # x_a = x[a, :]
            x_b = x[b, :]
            term1[j] += (f(x_b) - y[b]) \
                        * x[a, i] * x[b, i] * x[a, j]
    term1 *= 2 / n**2

    # hessians = np.sum(x * x, axis=0) / n
    tr_hessian = np.sum(x * x) / n

    # TensorFlow
    with tf.Session() as sess:
      x_t = tf.placeholder(tf.float32, shape=[None, d])
      y_t = tf.placeholder(tf.float32, shape=[None])

      w_t = tf.Variable(w, name='w')

      sess.run(tf.global_variables_initializer())

      logits_t = tf.einsum('ai,i->a', x_t, w_t)
      loss_t = tf.norm(logits_t - y_t)**2 / (2 * n)

      grad_t = tf.gradients(loss_t, w_t)
      grad_norm_sqr_t = tf.reduce_sum([tf.reduce_sum(g * g) for g in grad_t])
      term1_t = tf.gradients(grad_norm_sqr_t, w_t)

      # This gives the whole Hessian matrix
      hessians_t = tf.hessians(loss_t, w_t)
      hessians_t = hessians_t[0]
      tr_hessian_t = tf.trace(hessians_t)

      #############################################################
      # Compute trace of Hessian in a way that scales to non-vector
      # weights
      tr_hessian_2_t = tfutils.trace_hessian(loss_t, logits_t, w_t)

      ##############################################################

      results = sess.run((loss_t, grad_t, grad_norm_sqr_t, term1_t,
                          tr_hessian_t, tr_hessian_2_t),
                         feed_dict={
                             x_t: x,
                             y_t: y
                         })

      loss_result, grad_result, grad_norm_sqr_result, \
          term1_result, tr_hessian_result, \
          tr_hessian_2_result = results

      self.assertAlmostEqual(loss, loss_result, precision)
      self.assertAlmostEqual(grad_norm_sqr, grad_norm_sqr_result, precision)

      for term1_elem, term1_result_elem in zip(term1, term1_result[0]):
        self.assertAlmostEqual(term1_elem, term1_result_elem, precision)

      self.assertAlmostEqual(tr_hessian, tr_hessian_result, precision)
      self.assertAlmostEqual(tr_hessian, tr_hessian_2_result, precision)

      for g_elem, g_result_elem in zip(grad, grad_result[0]):
        self.assertAlmostEqual(g_elem, g_result_elem, precision)

  def _compute_hess_grad(self, L_t, w_t):
    grad_t = tf.gradients(L_t, w_t)
    grad_norm_sqr_t = tf.reduce_sum([tf.reduce_sum(g * g) for g in grad_t])
    return tf.gradients(grad_norm_sqr_t / 2., w_t)

  """Test calculation of derivative terms in the Fokker-Planck

    equation.
  """

  def test_term1(self):
    num_samples = 20
    dim = 10
    x = np.random.randn(num_samples, dim).astype(np.float32)
    y = np.random.randn(num_samples).astype(np.float32)
    w = np.random.randn(dim).astype(np.float32)

    f = np.dot(x, w)

    x_mat = np.matmul(x, x.transpose())
    term1 = np.dot(np.dot(x_mat, f - y), x) * (2. / num_samples**2)

    x_t = tf.constant(x)
    y_t = tf.constant(y)
    w_t = tf.Variable(w)
    f_t = tf.einsum('ai,i->a', x_t, w_t)
    L_t = tf.reduce_mean(tf.square(y_t - f_t)) / 2.

    term1_t = self._compute_hess_grad(L_t, w_t)

    # self.assertEqual('foo'.upper(), 'FOO')

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      term1_actual = sess.run(term1_t)[0] * 2.
      self.assertTrue(np.allclose(term1_actual, term1))

  def test_hessian_gradient(self):
    dim = 10
    w = np.random.randn(dim).astype(np.float32)
    w_t = tf.Variable(w)
    w_normsqr_t = tf.reduce_sum(w_t * w_t)
    L_t = w_normsqr_t + 3. * w_normsqr_t * w_normsqr_t \
          - 2.5 * w_normsqr_t * w_normsqr_t * w_normsqr_t

    grad_t = tf.gradients(L_t, w_t)[0]
    hess_t = tf.hessians(L_t, w_t)[0]

    expected_Hg_t = tf.einsum('ij,j->i', hess_t, grad_t)

    actual_Hg_t = self._compute_hess_grad(L_t, w_t)[0]

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      expected_Hg = sess.run(expected_Hg_t)
      actual_Hg = sess.run(actual_Hg_t)
      self.assertTrue(np.allclose(expected_Hg, actual_Hg))

  def test_hessian_gradient_2(self):
    dim = 10
    w1_t = tf.Variable(np.random.randn(dim).astype(np.float32))
    w2_t = tf.Variable(np.random.randn(dim).astype(np.float32))

    w1w1_t = tf.reduce_sum(w1_t * w1_t)
    w1w2_t = tf.reduce_sum(w1_t * w2_t)
    w2w2_t = tf.reduce_sum(w2_t * w2_t)

    L_t = 0.3 * w1w1_t + 0.1 * w1w2_t - 0.2 * w2w2_t \
          + 0.15 * w1w1_t * w1w1_t \
          - 0.45 * w1w1_t * w2w2_t \
          + 0.23 * w1w2_t * w1w1_t

    grad_t = tf.gradients(L_t, [w1_t, w2_t])
    H11_t = tf.hessians(L_t, w1_t)[0]
    H22_t = tf.hessians(L_t, w2_t)[0]
    H12_t = [tf.gradients(grad_t[0][i], w2_t)[0] for i in range(dim)]
    H21_t = [tf.gradients(grad_t[1][i], w1_t)[0] for i in range(dim)]

    actual_Hg_t = self._compute_hess_grad(L_t, [w1_t, w2_t])

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      grads = sess.run(grad_t)
      H11 = sess.run(H11_t)
      H22 = sess.run(H22_t)
      H12 = np.stack(sess.run(H12_t))
      H21 = np.stack(sess.run(H21_t))

      H = np.zeros((2 * dim, 2 * dim))
      H[:dim, :dim] = H11
      H[dim:, dim:] = H22
      H[:dim, dim:] = H12
      H[dim:, :dim] = H21

      grad = np.zeros(2 * dim)
      grad[:dim] = grads[0]
      grad[dim:] = grads[1]

      expected_Hg = H.dot(grad)
      actual_Hg = np.concatenate(sess.run(actual_Hg_t))

      self.assertTrue(np.allclose(expected_Hg, actual_Hg, rtol=1e-3))

  def test_full_hessian(self):
    dim1 = 10
    dim2 = 15

    w1_t = tf.Variable(np.random.randn(dim1).astype(np.float32))
    w2_t = tf.Variable(np.random.randn(dim2).astype(np.float32))

    w1w1_t = tf.reduce_sum(w1_t * w1_t)
    w2w2_t = tf.reduce_sum(w2_t * w2_t)

    L_t = 0.3 * w1w1_t - 0.2 * w2w2_t \
          + 0.15 * w1w1_t * w1w1_t - 0.45 * w1w1_t * w2w2_t

    grad_t = tf.gradients(L_t, [w1_t, w2_t])
    H11_t = tf.hessians(L_t, w1_t)[0]
    H22_t = tf.hessians(L_t, w2_t)[0]
    H12_t = [tf.gradients(grad_t[0][i], w2_t)[0] for i in range(dim1)]
    H21_t = [tf.gradients(grad_t[1][i], w1_t)[0] for i in range(dim2)]

    hess_blocks_t = tfutils.hessian_tensor_blocks(L_t, [w1_t, w2_t])

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      H11 = sess.run(H11_t)
      H22 = sess.run(H22_t)
      H12 = np.stack(sess.run(H12_t))
      H21 = np.stack(sess.run(H21_t))

      H = np.zeros((dim1 + dim2, dim1 + dim2))
      H[:dim1, :dim1] = H11
      H[dim1:, dim1:] = H22
      H[:dim1, dim1:] = H12
      H[dim1:, :dim1] = H21

      hess_blocks = sess.run(hess_blocks_t)

    actual_hess = tfutils.hessian_combine_blocks(hess_blocks)
    self.assertTrue(np.allclose(actual_hess, H))

  def test_hessian_vector_product(self):
    dim = 5
    w = np.random.randn(dim).astype(np.float32)
    w_t = tf.Variable(w)
    w_normsqr_t = tf.reduce_sum(w_t * w_t)
    L_t = w_normsqr_t + 3. * w_normsqr_t * w_normsqr_t \
          - 2.5 * w_normsqr_t * w_normsqr_t * w_normsqr_t

    v = np.random.randn(dim).astype(np.float32)
    v_t = tf.Variable(v)

    hess_t = tf.hessians(L_t, w_t)[0]

    expected_Hv_t = tf.einsum('ij,j->i', hess_t, v_t)
    actual_Hv_t = tfutils.hessian_vector_product(L_t, w_t, v_t)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      expected_Hv = sess.run(expected_Hv_t)
      actual_Hv = sess.run(actual_Hv_t)
      self.assertTrue(np.allclose(expected_Hv, actual_Hv))

  def test_hessian_spectrum_lanczos(self):
    K.clear_session()

    n = 10
    p = 4
    x = np.random.rand(n).astype(np.float32)
    y = np.sin(2 * np.pi * x).reshape((-1, 1)).astype(np.float32)

    features = np.zeros((n, p)).astype(np.float32)
    for order in range(p):
      features[:, order] = np.power(x, order)

    # Linear regression
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, use_bias=False, input_shape=(p,)))
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD())

    hess_t = tfutils.hessians(model.total_loss, model.trainable_weights[0])[0]

    hess = tfutils.keras_compute_tensors(model, features, y, hess_t).reshape(
        p, p)
    evals, evecs = np.linalg.eigh(hess)

    spec = tfutils.KerasHessianSpectrum(model, features, y)
    actual_evals, actual_evecs = spec.compute_spectrum(k=p - 1)

    self.assertTrue(np.allclose(evals[1:], actual_evals, rtol=1e-3))

    for i in range(p - 1):
      vec = evecs[:, i + 1]
      actual_vec = actual_evecs[:, i]
      self.assertTrue(
          np.allclose(vec, actual_vec, rtol=1e-3) or
          np.allclose(vec, -actual_vec, rtol=1e-3))

  def test_hessian_spectrum(self):
    K.clear_session()

    n = 10
    p = 3
    x = np.random.rand(n).astype(np.float32)
    y = np.sin(2 * np.pi * x).reshape((-1, 1)).astype(np.float32)

    features = np.zeros((n, p)).astype(np.float32)
    for order in range(p):
      features[:, order] = np.power(x, order)

    # Linear regression
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, use_bias=False, input_shape=(p,)))
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD())

    hess_t = tfutils.hessians(model.total_loss, model.trainable_weights[0])[0]

    hess = tfutils.keras_compute_tensors(model, features, y, hess_t).reshape(
        p, p)
    evals, evecs = np.linalg.eigh(hess)
    leading_eval = evals[-1]
    leading_evec = evecs[:, -1]

    spec = tfutils.KerasHessianSpectrum(model, features, y)
    actual_eval, actual_evec = spec.compute_leading_ev(epsilon=1e-4)

    self.assertTrue(np.isclose(leading_eval, actual_eval, rtol=1e-3))
    self.assertTrue(
        np.allclose(leading_evec, actual_evec, rtol=1e-3) or
        np.allclose(leading_evec, -actual_evec, rtol=1e-3))

    # Test other edge
    actual_other_edge, actual_evec = spec.compute_other_edge(
        leading_ev=actual_eval, epsilon=1e-5)
    self.assertTrue(np.isclose(evals[0], actual_other_edge, rtol=1e-3))
    self.assertTrue(
        np.allclose(evecs[:, 0], actual_evec, rtol=1e-3) or
        np.allclose(evecs[:, 0], -actual_evec, rtol=1e-3))

    # Run the same test with -loss, so the leading eigenvalue is
    # negative.
    spec = tfutils.KerasHessianSpectrum(
        model, features, y, loss=-model.total_loss)
    actual_eval, actual_evec = spec.compute_leading_ev(epsilon=1e-4)

    self.assertTrue(np.isclose(-leading_eval, actual_eval, rtol=1e-3))
    self.assertTrue(
        np.allclose(leading_evec, actual_evec, rtol=1e-3) or
        np.allclose(leading_evec, -actual_evec, rtol=1e-3))

  def test_hessian_spectrum_with_matrix_vector_action(self):
    """Test finding the leading eigenvalue of (1 - eta * H)."""
    K.clear_session()

    n = 10
    p = 3
    eta = 0.7

    x = np.random.rand(n).astype(np.float32)
    y = np.sin(2 * np.pi * x).reshape((-1, 1)).astype(np.float32)

    features = np.zeros((n, p)).astype(np.float32)
    for order in range(p):
      features[:, order] = np.power(x, order)

    # Linear regression
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, use_bias=False, input_shape=(p,)))
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD())

    hess_t = tfutils.hessians(model.total_loss, model.trainable_weights[0])[0]

    hess = tfutils.keras_compute_tensors(model, features, y, hess_t).reshape(
        p, p)
    A = np.identity(p) - eta * hess
    evals, evecs = np.linalg.eigh(A)

    if np.abs(evals[0]) > np.abs(evals[-1]):
      leading_eval = evals[0]
      leading_evec = evecs[:, 0]
    else:
      leading_eval = evals[-1]
      leading_evec = evecs[:, -1]

    spec = tfutils.KerasHessianSpectrum(model, features, y)
    actual_eval, actual_evec = spec.compute_leading_ev(
        epsilon=1e-5, matrix_vector_action=lambda v, Hv: v - eta * Hv)

    self.assertTrue(np.isclose(leading_eval, actual_eval, rtol=1e-3))
    self.assertTrue(
        np.allclose(leading_evec, actual_evec, rtol=1e-2) or
        np.allclose(leading_evec, -actual_evec, rtol=1e-2))

  def test_hessian_spectrum_batch_size_independence(self):
    K.clear_session()

    num_samples = 4096
    x = np.random.rand(num_samples).reshape((-1, 1))
    y = np.sin(2 * np.pi * x)

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(100, input_shape=(1,)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(100))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD())

    spec1 = tfutils.KerasHessianSpectrum(model, x, y, batch_size=32)
    spec2 = tfutils.KerasHessianSpectrum(model, x, y, batch_size=1024)

    ev1, _ = spec1.compute_leading_ev()
    ev2, _ = spec2.compute_leading_ev()
    self.assertTrue(np.isclose(ev1, ev2))

  def test_lanczos_eigsh(self):
    n = 10
    dtype = np.float32
    A = np.random.randn(n, n).astype(dtype)
    A = (A + A.transpose()) / 2

    # full_w, full_v = np.linalg.eigh(A)

    k = 6
    expected_w, expected_v = scipy.sparse.linalg.eigsh(A, k)

    def matvec(x):
      return A.dot(x)

    actual_w, actual_v = lanczos.eigsh(n, dtype, matvec)

    self.assertTrue(np.allclose(expected_w, actual_w))

    for i in range(k):
      exp_v = expected_v[:, i]
      act_v = actual_v[:, i]
      rtol = 1e-3
      self.assertTrue(
          np.allclose(exp_v, act_v, rtol=rtol) or
          np.allclose(exp_v, -act_v, rtol=rtol))

  def test_compute_sample_mean_tensor(self):
    K.clear_session()

    d = 12
    n = 20
    batch_size = n // 4
    x = np.random.rand(n, d).astype(np.float32)
    y = np.sin(2 * np.pi * x[:, 0]).reshape((-1, 1)).astype(np.float32)

    # Linear regression
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, use_bias=False, input_shape=(d,)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(10))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD())

    tfutils.keras_compute_tensors(model, x, y, model.total_loss)

    grad_t = tfutils.flatten_tensor_list(
        tf.gradients(model.total_loss, model.trainable_weights))
    grad = tfutils.keras_compute_tensors(model, x, y, grad_t)

    batches = tfutils.MiniBatchMaker(x, y, batch_size)

    actual_grad = tfutils.compute_sample_mean_tensor(model, batches, grad_t)

    self.assertTrue(np.allclose(grad, actual_grad))


class MockRecorder:
  def log_and_summary(*params):
    pass

  def record_scalar(*params):
    pass


class TestMeasurements(unittest.TestCase):

  def test_gradient_measurement(self):
    """Test that the full-batch gradient is computed correctly."""
    K.clear_session()

    d = 12
    n = 20
    batch_size = n // 4
    x = np.random.rand(n, d).astype(np.float32)
    y = np.sin(2 * np.pi * x[:, 0]).reshape((-1, 1)).astype(np.float32)

    x_test = np.random.rand(n, d).astype(np.float32)
    y_test = np.sin(2 * np.pi * x_test[:, 0]).reshape((-1,
                                                       1)).astype(np.float32)

    # Linear regression
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, use_bias=False, input_shape=(d,)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(10))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD())

    tfutils.keras_compute_tensors(model, x, y, model.total_loss)

    grad_t = tfutils.flatten_tensor_list(
        tf.gradients(model.total_loss, model.trainable_weights))
    grad = tfutils.keras_compute_tensors(model, x, y, grad_t)

    train_batches = tfutils.MiniBatchMaker(x, y, batch_size)
    test_batches = tfutils.MiniBatchMaker(x_test, y_test, batch_size)

    meas = measurements.GradientMeasurement(
        MockRecorder(), model,
        measurements.Frequency(freq=1, stepwise=False),
        train_batches, test_batches)

    meas.on_epoch_begin(0)
    meas.on_batch_begin(0)
    meas.on_batch_end(0)
    meas.on_epoch_end(0)
    actual_grad = meas.full_batch_g
    self.assertTrue(np.allclose(grad, actual_grad))

  def test_full_hessian_measurement(self):
    """Test that the Hessian is computed correctly."""
    K.clear_session()

    n = 10
    p = 4
    x = np.random.rand(n).astype(np.float32)
    y = np.sin(2 * np.pi * x).reshape((-1, 1)).astype(np.float32)

    features = np.zeros((n, p)).astype(np.float32)
    for order in range(p):
      features[:, order] = np.power(x, order)

    # Linear regression
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, use_bias=False, input_shape=(p,)))
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD())

    hess_t = tfutils.hessians(model.total_loss, model.trainable_weights[0])[0]

    hess = tfutils.keras_compute_tensors(model, features, y, hess_t).reshape(
        p, p)

    batch_size = n // 4
    batches = tfutils.MiniBatchMaker(features, y, batch_size)
    meas = measurements.FullHessianMeasurement(MockRecorder(), model, 1, batches,
                                               None, 1)
    actual_hess = meas.compute_hessian()

    self.assertTrue(np.allclose(hess, actual_hess))
    self.assertFalse(np.allclose(hess, 2 * actual_hess))


def main(_):
  unittest.main()


if __name__ == '__main__':
  tf.app.run(main)
