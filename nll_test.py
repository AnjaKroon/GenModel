from matplotlib import pyplot as plt
import scipy
from sklearn.covariance import log_likelihood
from file_helper import create_prefix_from_list, load_samples, store_for_plotting
from plotting import get_ci, get_color
from sampling.loading_samples import load_generative_model_samples
from sampling.stair import make_stair_prob
from sampling.discrete import makeUniProbArr, prob_array_to_dict
from statistic.binning_algo import binning_on_samples
from statistic.generate_statistics import genSstat, get_pmf_val, get_ranking_results, reject_if_bad_test
import numpy as np
import random
import math
from tqdm import tqdm


if __name__ == '__main__':
    # Set the random seed
    np.random.seed(3)
    random.seed(3)
    print('You are running the synthetic experiment...')

    power_base = 10
    U = power_base**power_base
    Ms = [100000,  1000000]
    TYPE = "TAIL" # sharp, flat, uniform, anom
    # power_base = 6
    # U = power_base**power_base
    # Ms = [10, 20,50,100]d

    init_e = 0.01
    init_b = 0.5
    trials = 5
    print(['{:.6f}'.format((m/U)*100 * trials) for m in Ms])

    S = 2
    ratio = 2
    distribution_type = 'STAIRS'  # STAIRS
    list_of_espilon_q = [0,init_e, init_e*2, init_e*4]
    list_of_title_q = ['zero','0.05', '0.1', '0.2']

    if distribution_type == 'UNIFORM':
        ground_truth_p = prob_array_to_dict(makeUniProbArr(U))

    elif distribution_type == 'STAIRS':
        ground_truth_p = make_stair_prob(U, posU=1, ratio=ratio,  S=S)

    else:
        raise NotImplemented
    ratios = []
    all_log_likelihoods_baselines_ratios = []
    for m in Ms:
        ratio = m/U
        ratios.append(ratio)
        print('Ration m/Omega', m/U)
        list_of_samples, list_of_pmf_q = load_samples(
            list_of_espilon_q, init_b, ground_truth_p, trials, U, m, S, ratio, TYPE)
        print('computing exact log likelihood...')
        all_samples_list = list_of_samples[0]
        all_log_likelihoods_baselines = []
        for i, q_name in enumerate(list_of_title_q):
            log_likelihoods = []
            pmf = list_of_pmf_q[i]
            q_name = list_of_title_q[i]

            for trial in all_samples_list:
                log_likelihood = 0
                for key, val in trial.items():
                    p_key = get_pmf_val(key, pmf)
                    log_p = np.log(p_key)
                    num_int = int(val * m)
                    log_likelihood += log_p * num_int
                log_likelihoods.append(-log_likelihood/m)
            print(q_name, 'log likelihood m=', m, ':', np.mean(
                log_likelihoods), 'std', np.std(log_likelihoods))

            all_log_likelihoods_baselines.append(log_likelihoods)
        all_log_likelihoods_baselines_ratios.append(
            all_log_likelihoods_baselines)
    for i, q_name in enumerate(list_of_title_q):

        key = q_name
        color = get_color(i)

        all_trials = val
        y = [np.mean(all_log_likelihoods_baselines[i])
             for all_log_likelihoods_baselines in all_log_likelihoods_baselines_ratios]

        plt.plot(ratios, y, color=color, label=key)
        try:
            all_ci = [get_ci(all_log_likelihoods_baselines[i])
                      for all_log_likelihoods_baselines in all_log_likelihoods_baselines_ratios]
            ci_over = [ci.high for i, ci in enumerate(all_ci)]
            ci_under = [ci.low for i, ci in enumerate(all_ci)]
            plt.fill_between(ratios, ci_under, ci_over, color=color, alpha=.1)
        except:
            print('couldnt do the ci')
    plt.legend()
    plt.xticks(ratios, ['{:.6f}'.format(r*100 * trials) for r in ratios])
    plt.show()
