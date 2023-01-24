from sklearn.covariance import log_likelihood
from file_helper import create_prefix_from_list, load_samples, store_for_plotting
from sampling.loading_samples import load_generative_model_samples
from sampling.stair import make_stair_prob
from sampling.discrete import makeUniProbArr, prob_array_to_dict
from statistic.binning_algo import binning_on_samples
from statistic.generate_statistics import genSstat, get_ranking_results, reject_if_bad_test
import numpy as np
import random
import math
from tqdm import tqdm


if __name__ == '__main__':
    # Set the random seed
    np.random.seed(3)
    random.seed(3)
    experiment = "SYNTH"  # either SYNTH or GEN
    test_epsilon = 0.1
    delta = 0.05
    compute_random = False
    list_of_binning_algo = ['algo']
    if experiment == "SYNTH":  # if we generate q ourselves
        print('You are running the synthetic experiment...')

        power_base = 6
        U = power_base**power_base
        m = 100000
        init_e = 0.05
        init_b = 100
        trials = 10
        S = 3
        ratio = 2
        distribution_type = 'STAIRS'  # STAIRS
        list_of_espilon_q = [0, init_e, init_e*2, init_e*4]
        list_of_title_q = [
            'no temper (uniform)', 'slightly tempered', 'medium tempered', 'heavily tempered']

    else:  # if we take q as the generative models we have, we load the samples.
        print('You are running the generative model experiment...')
        power_base = 6
        U = power_base**power_base
        m = 10000
        S = 2
        ratio = 3
        trials = 10

    Bs = list(range(S+1, 2*(S+1)+1))

    print("for this round m is ", m)
    print("and U is ", U)

    prefix = create_prefix_from_list([experiment, U, m, trials, S, ratio])
    if experiment == "SYNTH":
        if distribution_type == 'UNIFORM':
            ground_truth_p = prob_array_to_dict(makeUniProbArr(U))

        elif distribution_type == 'STAIRS':
            ground_truth_p = make_stair_prob(
                U, posU=(math.factorial(power_base)/U), ratio=ratio,  S=S)

        else:
            raise NotImplemented
        list_of_samples, list_of_pmf_q = load_samples(
            list_of_espilon_q, init_b, ground_truth_p, trials, U, m, S, ratio)
        print('computing exact log likelihood...')
        all_samples_list = list_of_samples[0]
        log_likelihoods = []
        for i, q_name in enumerate(list_of_title_q):
            pmf = list_of_pmf_q[i]
            q_name = list_of_title_q[i]

            for trial in all_samples_list:
                log_likelihood = 0
                for key, val in trial.items():
                    log_p = np.log(pmf[key])
                    num_int = int(val * m)
                    log_likelihood += log_p * num_int
                log_likelihoods.append(log_likelihood/m)
            print(q_name, 'log likelihood m=', m, ':', np.mean(
                log_likelihoods), 'std', np.std(log_likelihoods))
    else:
        dict_of_samples, ground_truth_p = load_generative_model_samples(
            power_base, num_files=2)
        list_of_samples = [val for _, val in dict_of_samples.items()]
        list_of_title_q = [key for key, _ in dict_of_samples.items()]

    store_results_algo = {}
    store_results_random = {}
    store_results_ranking = {}
    for algo in list_of_binning_algo:
        store_results_ranking[algo] = []
    metrics = ['S', 'test', 'binning']
    for metric in metrics:
        store_results_algo[metric] = {}
        store_results_random[metric] = {}
        for title in list_of_title_q:
            store_results_algo[metric][title] = []
            store_results_random[metric][title] = []

    for B in tqdm(Bs):  # For each bin granularity

        for i, all_samples_list in enumerate(list_of_samples):
            list_binned_algo = binning_on_samples(
                'algo', all_samples_list, ground_truth_p, U, B)
            # run statistical test
            test_algo = [reject_if_bad_test(
                trial['p'], trial['q'], m, epsilon=test_epsilon, delta=delta)[0] for trial in list_binned_algo]

            # compute S reults
            S_algo = [genSstat(trial['p'], trial['q'])
                      for trial in list_binned_algo]
            if compute_random:
                list_binned_random = binning_on_samples(
                    'random', all_samples_list, ground_truth_p, U, B)
                test_random = [reject_if_bad_test(
                    trial['p'], trial['q'], m, epsilon=test_epsilon, delta=delta) for trial in list_binned_random]
                S_random = [genSstat(trial['p'], trial['q'])
                            for trial in list_binned_random]

            q_name = list_of_title_q[i]

            store_results_algo['test'][q_name].append(test_algo)
            store_results_algo['S'][q_name].append(S_algo)
            store_results_algo['binning'][q_name].append(list_binned_algo)
            if compute_random:
                store_results_random['test'][q_name].append(
                    test_random)
                store_results_random['S'][q_name].append(S_random)
        # compute correct correct with S
        ranking_algo = get_ranking_results(
            [store_results_algo['S'][q_name][-1] for q_name in list_of_title_q])
        store_results_ranking['algo'].append(ranking_algo)
        if compute_random:
            ranking_random = get_ranking_results(
                [store_results_random['S'][q_name][-1] for q_name in list_of_title_q])
            store_results_ranking['random'].append(ranking_random)

    store_for_plotting(
        data={'x': Bs, 'data': store_results_algo['binning']}, title=prefix+'_binning_algo')
    store_for_plotting(
        data={'x': Bs, 'data': store_results_algo['test']}, title=prefix+'_hypothesis_algo')
    store_for_plotting(
        data={'x': Bs, 'data': store_results_algo['S']}, title=prefix+'_S_algo')
    label_dict = {'algo': r'$\mathcal{B}^*_k$',
                  'random': r'random  $\mathcal{B}_k$'}
    store_for_plotting(
        data={'x': Bs, 'data': store_results_ranking, 'label_dict': label_dict}, title=prefix+'_ranking')
