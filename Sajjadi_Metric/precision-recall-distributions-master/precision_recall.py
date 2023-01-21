# coding=utf-8
# Taken from:
# https://github.com/google/compare_gan/blob/master/compare_gan/src/prd_score.py
#
# Changes:
#   - default dpi changed from 150 to 300
#   - added handling of cases where P = Q, where precision/recall may be
#     just above 1, leading to errors for the f_beta computation
#
# Copyright 2018 Google LLC & Hwalsuk Lee.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Precision and recall computation based on samples from two distributions.

Given a sample from the true and the fake distribution embedded in some feature
space (say, Inception), it computes the precision and recall via the algorithm
presented in [arxiv.org/abs/1806.00035]. Finally, one can plot the resulting
curves for different models.

Typical usage example:

import prd
  prd_data_1 = prd.compute_prd_from_embedding(eval_feats_1, ref_feats_1)
  prd_data_2 = prd.compute_prd_from_embedding(eval_feats_2, ref_feats_2)
  prd.plot([prd_data_1, prd_data_2], ['GAN_1', 'GAN_2'])
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib import pyplot as plt
import numpy as np
import sklearn.cluster
import pickle as pkl

def make_eval_ref_dist():
    infile = open('100sample.pk','rb')
    eval_dist = pkl.load(infile)
    ref_dist = np.ones((100000, 6), int)
    print("should be equal", eval_dist.shape, ref_dist.shape)
    # something with the dimensions is off. Going to do a toy example

    eval_dist = [0,1,2,3,4,5]
    ref_dist = [0,1,2,4,4,5]
    # ValueError: Detected value > 1.001, this should not happen.

    eval_dist = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ref_dist = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # ValueError: Detected value > 1.001, this should not happen.

    eval_dist = [0,1]
    ref_dist = [0,1]
    # this copies now the example in prd_score_test

    return eval_dist, ref_dist

def compute_prd(eval_dist, ref_dist, num_angles=3, epsilon=1e-10):
  """Computes the PRD curve for discrete distributions.

  This function computes the PRD curve for the discrete distribution eval_dist
  with respect to the reference distribution ref_dist. This implements the
  algorithm in [arxiv.org/abs/1806.2281349]. The PRD will be computed for an
  equiangular grid of num_angles values between [0, pi/2].

  Args:
    eval_dist: 1D NumPy array or list of floats with the probabilities of the
               different states under the distribution to be evaluated.
    ref_dist: 1D NumPy array or list of floats with the probabilities of the
              different states under the reference distribution.
    num_angles: Number of angles for which to compute PRD. Must be in [3, 1e6].
                The default value is 1001.
    epsilon: Angle for PRD computation in the edge cases 0 and pi/2. The PRD
             will be computes for epsilon and pi/2-epsilon, respectively.
             The default value is 1e-10.

  Returns:
    precision: NumPy array of shape [num_angles] with the precision for the
               different ratios.
    recall: NumPy array of shape [num_angles] with the recall for the different
            ratios.

  Raises:
    ValueError: If not 0 < epsilon <= 0.1.
    ValueError: If num_angles < 3.
  """

  if not (epsilon > 0 and epsilon < 0.1):
    raise ValueError('epsilon must be in (0, 0.1] but is %s.' % str(epsilon))
  if not (num_angles >= 3 and num_angles <= 1e6):
    raise ValueError('num_angles must be in [3, 1e6] but is %d.' % num_angles)

  # Compute slopes for linearly spaced angles between [0, pi/2]
  angles = np.linspace(epsilon, np.pi/2 - epsilon, num=num_angles)
  slopes = np.tan(angles)

  # Broadcast slopes so that second dimension will be states of the distribution
  slopes_2d = np.expand_dims(slopes, 1)

  # Broadcast distributions so that first dimension represents the angles
  ref_dist_2d = np.expand_dims(ref_dist, 0)
  eval_dist_2d = np.expand_dims(eval_dist, 0)

  # Compute precision and recall for all angles in one step via broadcasting
  precision = np.minimum(ref_dist_2d*slopes_2d, eval_dist_2d).sum(axis=1)
  recall = precision / slopes

  # handle numerical instabilities leaing to precision/recall just above 1
  max_val = max(np.max(precision), np.max(recall))
  if max_val > 1.001:
    raise ValueError('Detected value > 1.001, this should not happen.')
  precision = np.clip(precision, 0, 1)
  recall = np.clip(recall, 0, 1)

  return precision, recall


def _prd_to_f_beta(precision, recall, beta=1, epsilon=1e-10):
  """Computes F_beta scores for the given precision/recall values.

  The F_beta scores for all precision/recall pairs will be computed and
  returned.

  For precision p and recall r, the F_beta score is defined as:
  F_beta = (1 + beta^2) * (p * r) / ((beta^2 * p) + r)

  Args:
    precision: 1D NumPy array of precision values in [0, 1].
    recall: 1D NumPy array of precision values in [0, 1].
    beta: Beta parameter. Must be positive. The default value is 1.
    epsilon: Small constant to avoid numerical instability caused by division
             by 0 when precision and recall are close to zero.

  Returns:
    NumPy array of same shape as precision and recall with the F_beta scores for
    each pair of precision/recall.

  Raises:
    ValueError: If any value in precision or recall is outside of [0, 1].
    ValueError: If beta is not positive.
  """

  if not ((precision >= 0).all() and (precision <= 1).all()):
    raise ValueError('All values in precision must be in [0, 1].')
  if not ((recall >= 0).all() and (recall <= 1).all()):
    raise ValueError('All values in recall must be in [0, 1].')
  if beta <= 0:
    raise ValueError('Given parameter beta %s must be positive.' % str(beta))

  return (1 + beta**2) * (precision * recall) / (
      (beta**2 * precision) + recall + epsilon)


def prd_to_max_f_beta_pair(precision, recall, beta=8):
  """Computes max. F_beta and max. F_{1/beta} for precision/recall pairs.

  Computes the maximum F_beta and maximum F_{1/beta} score over all pairs of
  precision/recall values. This is useful to compress a PRD plot into a single
  pair of values which correlate with precision and recall.

  For precision p and recall r, the F_beta score is defined as:
  F_beta = (1 + beta^2) * (p * r) / ((beta^2 * p) + r)

  Args:
    precision: 1D NumPy array or list of precision values in [0, 1].
    recall: 1D NumPy array or list of precision values in [0, 1].
    beta: Beta parameter. Must be positive. The default value is 8.

  Returns:
    f_beta: Maximum F_beta score.
    f_beta_inv: Maximum F_{1/beta} score.

  Raises:
    ValueError: If beta is not positive.
  """

  if not ((precision >= 0).all() and (precision <= 1).all()):
    raise ValueError('All values in precision must be in [0, 1].')
  if not ((recall >= 0).all() and (recall <= 1).all()):
    raise ValueError('All values in recall must be in [0, 1].')
  if beta <= 0:
    raise ValueError('Given parameter beta %s must be positive.' % str(beta))

  f_beta = np.max(_prd_to_f_beta(precision, recall, beta))
  f_beta_inv = np.max(_prd_to_f_beta(precision, recall, 1/beta))
  return f_beta, f_beta_inv

if __name__ == '__main__':
    e,r = make_eval_ref_dist()
    p,r = compute_prd(e,r)
    print(p)
    print()
    print(r)
    
    #There are 1001 elements in the precision & recall metric.... why?