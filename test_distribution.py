import math
from tqdm import tqdm
from figure_generater import generating_S_rank_plots
from sampling.discrete import makeUniProbArr, prob_array_to_dict
from file_helper import load_samples
from sampling.stair import make_stair_prob
from statistic.binning_algo import binning_on_samples
import numpy as np
import random

from statistic.generate_statistics import reject_if_bad_test


if __name__ == '__main__':
    # Set the random seed
    np.random.seed(3)
    random.seed(3)

    init_e = 0.1
    init_b = 30
    trials = 50

    S = 3
    ratio = 2
    distribution_type = 'STAIRS'  # STAIRS

    Bs = [8]
    power_base = 6
    list_U = [power_base**power_base]
    list_M = [50000]

    list_of_binning_algo = ['algo', 'random']
    list_of_espilon_q = [ init_e*2]
    list_of_title_q = [
        'no temper (uniform)', 'slightly tempered', 'medium tempered', 'heavily tempered']

    # create list of results,
    list_of_results_stats = []
    for _ in list_of_binning_algo:
        list_of_q_results_stats = []
        for _ in list_of_espilon_q:
            list_of_q_results_stats.append([])
        list_of_results_stats.append(list_of_q_results_stats)

    for m in list_M:
        print("for this round m is ", m)
        for U in list_U:
            print("and U is ", U)

            if distribution_type == 'UNIFORM':
                ground_truth_p = prob_array_to_dict(makeUniProbArr(U))

            elif distribution_type == 'STAIRS':
                ground_truth_p = make_stair_prob(
                    U, posU=(math.factorial(power_base)/U), ratio=ratio,  S=S)

            else:
                raise NotImplemented
            list_of_samples = load_samples(
                list_of_espilon_q, init_b, ground_truth_p, trials, U, m)

            for B in tqdm(Bs):  # For each bin granularity

                for i, all_samples_list in enumerate(list_of_samples):
                    list_binned_algo = binning_on_samples(
                        'algo', all_samples_list, ground_truth_p, U, B)
                    list_binned_random = binning_on_samples(
                        'random', all_samples_list, ground_truth_p, U, B)
                    # test_algo = [reject_if_bad_test(
                    #     trial['p'], trial['q'], m) for trial in list_binned_algo]
                    test_random = [reject_if_bad_test(
                        trial['p'], trial['q'], m) for trial in list_binned_random]
                
                    print(np.mean(test_algo))
                    print(np.mean(test_random))
