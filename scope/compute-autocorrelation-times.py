#!/usr/bin/env python3

"""Compute autocorrelation times and other metrics from Lanczos results.

Usage: compute-autocorrelation-times.py [options]

Options:
  -h, --help            Show this.
  --logdir DIR          The experiment directory.
  --t1-step-size N      How many steps to skip when advancing t1 [Default: 50]
  --t2-step-size N      How many steps to skip when advancing t2 [Default: 10]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glob import glob
import re
import numpy as np
import pandas as pd
from scope.docopt_props import DocoptProperties

# log_dir = glob('logs/paperBasicOverlap-lanczos-*/lanczos_hessian')[0]
args = DocoptProperties(__doc__)
data_dir = args.logdir + '/lanczos_hessian/'
times = []

for fname in glob(data_dir + '/H_evecs_*.npy'):
    m = re.search(r'H_evecs_(\d+)', fname)
    t = m.group(1)
    times.append(t)

times_s = sorted(times, key=int)
times = [int(t) for t in times_s]


def load_file(t, name):
    return np.load('{}/{}_{:05d}.npy'.format(data_dir, name, t))


def subspace_overlap(V, V2, k=None):
    '''k = number of top vectors to use'''
    if k is None:
        V_k = V
        V2_k = V2
    else:
        V_k = V[:, -k:]
        V2_k = V2[:, -k:]

    prod_abs = np.abs(V2_k.transpose().dot(V_k))
    return np.mean(np.linalg.norm(prod_abs, axis=1)**2)


df = pd.DataFrame()
rand_df = pd.DataFrame()
classes = 10

for i, t in enumerate(times):
    evals = load_file(t, 'H_evals')
    top_evals = evals[-classes:]

    V = load_file(t, 'H_evecs')
    top_V = V[:, -classes:]
    next_V = V[:, -2*classes:-classes]

    rand_vec = np.random.randn(classes)
    rand_vec = rand_vec / np.linalg.norm(rand_vec)
    top_H_rand_vec = rand_vec * top_evals
    top_H_rand_vec = top_H_rand_vec / np.linalg.norm(top_H_rand_vec)
    rand_vec_top_overlap = rand_vec.dot(top_H_rand_vec)
    rand_df = rand_df.append(
        {'t': t, 'rand_vec_top_overlap': rand_vec_top_overlap},
        ignore_index=True)

    if i % args.t1_step_size != 0:
        continue

    for i2, t2 in enumerate(times[i:]):
        if i2 % args.t2_step_size != 0:
            continue

        V2 = load_file(t2, 'H_evecs')
        top_V2 = V2[:, -classes:]
        next_V2 = V2[:, -2*classes:-classes]

        # top_prod_abs = np.abs(top_V2.transpose().dot(top_V))
        # mean_self_overlap = np.mean(np.diag(top_prod_abs))
        # mean_max_overlap = np.mean(np.max(top_prod_abs, axis=1))
        # top_max_overlap = np.max(top_prod_abs[-1, :])

        topk_subspace_overlap = subspace_overlap(top_V, top_V2)
        top2_subspace_overlap = subspace_overlap(V, V2, k=2)
        top5_subspace_overlap = subspace_overlap(V, V2, k=5)
        top8_subspace_overlap = subspace_overlap(V, V2, k=9)
        top9_subspace_overlap = subspace_overlap(V, V2, k=9)
        top11_subspace_overlap = subspace_overlap(V, V2, k=11)
        next_subspace_overlap = subspace_overlap(next_V, next_V2)
        top2k_subspace_overlap = subspace_overlap(V, V2, k=2*classes)

        df = df.append({'t1': t, 't2': t2, 'dt': t2 - t,
                        # 'mean_self_overlap': mean_self_overlap,
                        # 'mean_max_overlap': mean_max_overlap,
                        # 'top_max_overlap': top_max_overlap,
                        'topk_subspace_overlap': topk_subspace_overlap,
                        'top2_subspace_overlap': top2_subspace_overlap,
                        'top5_subspace_overlap': top5_subspace_overlap,
                        'top8_subspace_overlap': top8_subspace_overlap,
                        'top9_subspace_overlap': top9_subspace_overlap,
                        'top11_subspace_overlap': top11_subspace_overlap,
                        'next_subspace_overlap': next_subspace_overlap,
                        'top2k_subspace_overlap': top2k_subspace_overlap},
                       ignore_index=True)
        print(t, t2, topk_subspace_overlap, next_subspace_overlap)

    df.to_pickle(data_dir + '/self_overlaps.pkl')

rand_df.to_pickle(data_dir + '/rand_vec_overlaps.pkl')
