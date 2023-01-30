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


def compute_pairwise_distance(data_x, data_y=None, distance=None): 
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """

    if data_y is None:
        data_y = data_x

    def get_category_bias(Z):
        """
        Args:
            Z: numpy.ndarray([N, feature_dim], dtype=np.float32) )
        Returns:
            all_cats: numpy.ndarray(category)
        """
        all_cats = []
        for row in range(Z.shape[0]):       # for each row in each array
            cat = 0                         # set category bias to 0
            permutation = False
            empty_list = []
            for each in Z[row, :]:
                if each not in empty_list: empty_list.append(each)
            if len(empty_list) == 6: permutation = True
            if Z[row, 0] > Z[row, -1] and permutation == True:
                cat = 1
            elif Z[row, 0] < Z[row, -1] and permutation == True:
                cat = 2
            if permutation == False: cat = 4
            # print(cat)
            all_cats.append(cat)
        return all_cats
    
    X_bias = np.array(get_category_bias(data_x))
    #print(X_bias[:10])
    Y_bias = np.array(get_category_bias(data_y))
    #print(Y_bias[:10])

    #X_bias = np.reshape(X_bias, (len(X_bias), 1))
    #Y_bias = np.reshape(Y_bias, (len(Y_bias), 1))
    #print(X_bias.shape)
    #print(Y_bias.shape)

    
    if distance is None:
        dists = sklearn.metrics.pairwise_distances(
            data_x, data_y, metric='euclidean', n_jobs=8)
    elif distance == 'hamilton':
        max_val = np.max([np.max(data_x), np.max(data_y)])
        one_hot_data_x = to_one_hot(data_x, max_val)
        one_hot_data_y = to_one_hot(data_y,max_val)
        # dists = sklearn.metrics.pairwise.manhattan_distances(one_hot_data_x, one_hot_data_y)
        dists_based_on_category = sklearn.metrics.pairwise.manhattan_distances(X_bias.reshape(-1,1), Y_bias.reshape(-1,1))
        dists_based_on_number = sklearn.metrics.pairwise.manhattan_distances(data_x, data_y)
        dists = dists_based_on_category + 0.001*dists_based_on_number
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

    distances = compute_pairwise_distance(input_features, distance=distance)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k=5, distance=None):
    """
    Computes precision, recall, density, and coverage given two manifolds.
    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """
    

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k, distance=distance)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k, distance=distance)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features, distance=distance)

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
