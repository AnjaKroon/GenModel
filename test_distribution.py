import math
from tqdm import tqdm
from plot_utils import plot_stat, put_on_plot
from sampling.discrete import makeUniProbArr, prob_array_to_dict
from file_helper import load_samples
from sampling.loading_samples import load_generative_model_samples
from sampling.stair import make_stair_prob
from statistic.binning_algo import binning_on_samples
import numpy as np
import random
from statistic.generate_statistics import reject_if_bad_test


if __name__ == '__main__':
    # Set the random seed
    np.random.seed(3)
    random.seed(3)
    experiment = "SYNTH"  # either SYNTH or GEN
    test_epsilon = 0.2
    delta = 1/3
    if experiment == "SYNTH":  # if we generate q ourselves
        print('You are running the synthetic experiment...')

        power_base = 6
        list_U = [power_base**power_base]
        list_M = [10000]
        init_e = 0.1
        init_b = 30
        trials = 50
        S = 3
        ratio = 2
        distribution_type = 'STAIRS'  # STAIRS

        list_of_espilon_q = [0, init_e, init_e*1.5, init_e*2]
        list_of_title_q = [
            'no temper (uniform)', 'slightly tempered', 'medium tempered', 'heavily tempered']
    else:  # if we take q as the generative models we have, we load the samples.
        print('You are running the generative model experiment...')
        power_base = 6
        list_U = [power_base**power_base]
        list_M = [10000]
        S = 2

    Bs = list(range(S+1, 2*(S+1)+1))
    list_of_binning_algo = ['algo', 'random']

    for m in list_M:
        print("for this round m is ", m)
        for U in list_U:
            print("and U is ", U)
            if experiment == "SYNTH":
                if distribution_type == 'UNIFORM':
                    ground_truth_p = prob_array_to_dict(makeUniProbArr(U))

                elif distribution_type == 'STAIRS':
                    ground_truth_p = make_stair_prob(
                        U, posU=(math.factorial(power_base)/U), ratio=ratio,  S=S)

                else:
                    raise NotImplemented
                list_of_samples = load_samples(
                    list_of_espilon_q, init_b, ground_truth_p, trials, U, m)
            else:
                dict_of_samples, ground_truth_p = load_generative_model_samples()
                list_of_samples = [val for _, val in dict_of_samples.items()]
                list_of_title_q = [key for key, _ in dict_of_samples.items()]

            store_results_algo = {}
            store_results_random = {}
            for title in list_of_title_q:
                store_results_algo[title] = []
                store_results_random[title] = []
            
            for B in tqdm(Bs):  # For each bin granularity

                for i, all_samples_list in enumerate(list_of_samples):
                    list_binned_algo = binning_on_samples(
                        'algo', all_samples_list, ground_truth_p, U, B)
                    list_binned_random = binning_on_samples(
                        'random', all_samples_list, ground_truth_p, U, B)
                    test_algo = [reject_if_bad_test(
                        trial['p'], trial['q'], m, epsilon=test_epsilon, delta=delta) for trial in list_binned_algo]
                    test_random = [reject_if_bad_test(
                        trial['p'], trial['q'], m, epsilon=test_epsilon, delta=delta) for trial in list_binned_random]

                    q_name = list_of_title_q[i]

                    store_results_algo[q_name].append(test_algo)
                    store_results_random[q_name].append(test_random)

    put_on_plot(Bs, store_results_algo)
    plot_stat(experiment+'algo.pdf', 'Bins', 'algo')

    put_on_plot(Bs, store_results_random)
    plot_stat(experiment+'random.pdf', 'Bins', 'random')
