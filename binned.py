#from discrete import makeUniProbArr, errFunct, genValArr, sampleSpecificProbDist
#from gen_S import empirical_dist, genSstat
#from plot_utils import plot_S_stat
#from sampling.poisson import poisson_empirical_dist
from decimal import ROUND_DOWN
from re import L
import sys
import numpy as np
import random
import math

# return the dict of a binned pmf, with predined binning mapping_from_index_to_bin
def p_to_bp_with_index(histo_p, U, B, mapping_from_index_to_bin):
    new_histo = {}
    for index, val in histo_p.items():
        bin = mapping_from_index_to_bin[index]
        if bin not in new_histo:
            new_histo[bin] = val
        else:
            new_histo[bin] += val
    return new_histo
def p_to_bp_random(histo_p, U, B):
    amount_per_bin = math.floor(U/B)  # 3
    amount_final_bin = int(amount_per_bin + (U % B))  # 4
    
    # shuffle the binning
    mapping_from_index_to_bin = {}
    mapping_bin_to_index = {}
    shuffled_U = list(range(U))
    random.shuffle(shuffled_U)
    for i in range(B):
        mapping_bin_to_index[i] = []
        size_bin = amount_per_bin
        if i == B-1:
            size_bin = amount_final_bin
        for j in range(size_bin):
            index = i * amount_per_bin + j
            shuffled_index = shuffled_U[index]
            mapping_from_index_to_bin[shuffled_index] = i
            mapping_bin_to_index[i].append(shuffled_index)
    new_histo = {}
    for bin_index, all_index in mapping_bin_to_index.items():
        new_probability_for_bin = 0
        for j in all_index:
            new_probability_for_bin = new_probability_for_bin + histo_p[j]
        new_histo[bin_index] = new_probability_for_bin
    return new_histo, mapping_from_index_to_bin


def transform_samples(b_p, histo_p, p_samples, U, B):
    # define subdivisions
    print("size of p_samples is ", len(p_samples))
    amount_per_bin = math.floor(len(p_samples)/B)
    amount_final_bin = amount_per_bin + (len(p_samples) % B)
    # bins = B
    new_samples = []
    # for item in histo_p.items():
    # if sample is from 0 to 33, add element [1] to new_samples
    # if sample is from 34 to 66, add element [2] to new_samples
    # if sample is from 67 to 100, add element [3] to new_samples
    for i in range(1, B+1):
        if i != B:
            for amt in range(amount_per_bin):
                new_samples.append(i)
        if i == B:
            for amt in range(amount_final_bin):
                new_samples.append(i)

    print(new_samples)

    # for i amount of bins
    # # i = 1 lets say
    # if not last bin, let's add numbers, for the amount_per_bin
    # if last bin, let's add numbers, for the amount_final_bin
    return new_samples


if __name__ == '__main__':
    sample_histo = {1: 0.15, 2: 0.15, 3: 0.1, 4: 0.05,
                    5: 0.1, 6: 0.1, 7: 0.05, 8: 0.1, 9: 0.1, 10: 0.1}
    sample_b = {1: 0.33, 2: 0.33, 3: 0.33}
    p_samples = [1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 4, 5, 6, 7, 8, 9, 10]

    U = 10
    B = 3

    b_p = p_to_bp_random(sample_histo, U, B)  # works
    print(b_p)
    print('probability sum to : ', sum(b_p.values()))
    b_out = transform_samples(b_p, sample_histo, p_samples, U, B)  # works

    # the question is how to get the relevant things in
    # need some sort of histo for the transform_samples -- there should be an array of the poissonized samples
