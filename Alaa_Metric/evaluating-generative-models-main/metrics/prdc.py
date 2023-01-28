# -*- coding: utf-8 -*-

"""
Taken from https://github.com/clovaai/generative-evaluation-prdc

prdc 
Copyright (c) 2020-present NAVER Corp.
MIT license

"""

import numpy as np
import sklearn.metrics

__all__ = ['compute_prdc']


def to_one_hot(x, max_val):
    flat_x = np.reshape(x, (x.shape[0]*x.shape[1]))
    b = np.zeros((x.shape[0] * x.shape[1], max_val+1))
    b[np.arange(flat_x.shape[0]), flat_x] = 1
    b = np.reshape(b, (x.shape[0], -1))
    return b


def compute_pairwise_distance(data_x, data_y=None, X_bias=None, Y_bias=None, distance=None): 
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    print("compute_pairwise_distance -- it's getting lost here")
    print(X_bias)
    print(Y_bias)


    if X_bias is None:
        X_bias = np.zeros(len(data_x))
    if Y_bias is None:
        Y_bias = np.zeros(len(data_x))
    if data_y is None:
        data_y = data_x
    if distance is None:
        dists = sklearn.metrics.pairwise_distances(
            data_x, data_y, metric='euclidean', n_jobs=8)
    elif distance == 'hamilton':
        max_val = np.max([np.max(data_x), np.max(data_y)])
        one_hot_data_x = to_one_hot(data_x, max_val)
        one_hot_data_y = to_one_hot(data_y,max_val)
        # dists = sklearn.metrics.pairwise.manhattan_distances(one_hot_data_x, one_hot_data_y)
        X_bias_loc = [X_bias.copy()]
        Y_bias_loc = [Y_bias.copy()]
        print(X_bias_loc)
        print(Y_bias_loc)
        dists = sklearn.metrics.pairwise.manhattan_distances(X_bias_loc, Y_bias_loc)
        # print(dists.shape)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k, distance):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features, X_bias=None, Y_bias=None, distance=distance)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, X_bias, Y_bias, nearest_k=5, distance=None):
    """
    Computes precision, recall, density, and coverage given two manifolds.
    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """
    print("compute_prdc")
    print(X_bias[:10])
    print(Y_bias[:10])

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k, distance=distance)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k, distance=distance)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features, X_bias, Y_bias, distance=distance)

    precision = (
        distance_real_fake <
        np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
        distance_real_fake <
        np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
        distance_real_fake <
        np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
        distance_real_fake.min(axis=1) <
        real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)
