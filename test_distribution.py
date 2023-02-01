import scipy
from sklearn.covariance import log_likelihood
from file_helper import create_prefix_from_list, load_samples, store_for_plotting
from sampling.loading_samples import load_generative_model_samples
from sampling.stair import make_stair_prob
from sampling.discrete import makeUniProbArr, prob_array_to_dict
from statistic.binning_algo import binning_on_samples
from statistic.generate_statistics import genSstat, get_pmf_val, get_ranking_results, reject_if_bad_test
import numpy as np
import random
import math
from tqdm import tqdm

from table_helper import build_latex_table


def compute_NLL(list_of_samples, list_of_title_q, store_results):
    all_samples_list = list_of_samples[0]
    all_log_likelihoods = []
    for i, q_name in enumerate(list_of_title_q):
        log_likelihoods = []
        pmf = list_of_pmf_q[i]
        q_name = list_of_title_q[i]

        for trial in all_samples_list:
            log_likelihood = 0
            for key, val in trial.items():
                p_key = get_pmf_val(key, pmf)
                log_p = np.log(p_key)
                num_int = int(val * m_per_splits)
                log_likelihood += log_p * num_int
            log_likelihoods.append(-log_likelihood/m_per_splits)
        print(q_name, 'log likelihood m=', m_per_splits, ':', np.mean(
            log_likelihoods), 'std', np.std(log_likelihoods))
        all_log_likelihoods.append(log_likelihoods)
        store_results['nll'][q_name] = log_likelihoods
    print(all_log_likelihoods[0])
    print(all_log_likelihoods[1])
    print(scipy.stats.wilcoxon(
        all_log_likelihoods[0], all_log_likelihoods[2]))


def consolidate(all_samples_list):
    sample_dict = {}
    num_splits = len(all_samples_list)
    for samples in all_samples_list:
        for key, emp_q in samples.items():
            if key not in sample_dict:
                sample_dict[key] = emp_q/num_splits
            else:
                sample_dict[key] += emp_q/num_splits
    print('should be one', np.sum(list(sample_dict.values())))
    return sample_dict


def perform_our_test(list_of_samples, list_of_title_q, S, trials, store_results):
    # step one consolidate all samples to one sample set
    consolidated_samples = []
    for all_samples_list in list_of_samples:
        consolidated_samples.append(consolidate(all_samples_list))

    Bs = list(range(S+1, 2*(S+1)+1))
    for B in tqdm(Bs):  # For each bin granularity

        for i, consolidated_samples_baseline in enumerate(consolidated_samples):

            list_binned = binning_on_samples(
                consolidated_samples_baseline, trials, ground_truth_p, U, B)
            # run statistical test
            results = [reject_if_bad_test(
                trial['p'], trial['q'], m_per_splits, epsilon=test_epsilon, delta=delta) for trial in list_binned]
            test = [i[0] for i in results]
            A = [i[1] for i in results]

            # compute S reults
            S = [genSstat(trial['p'], trial['q'])
                 for trial in list_binned]
            q_name = list_of_title_q[i]

            store_results['test'][q_name][B] = test
            store_results['A'][q_name][B] = A
            store_results['S'][q_name][B] = S
            store_results['binning'][q_name][B] = list_binned


if __name__ == '__main__':
    # Set the random seed
    np.random.seed(3)
    random.seed(3)
    experiment = "SYNTH"  # either SYNTH or GEN
    TYPE = "TAIL"  # sharp, flat, uniform, anom
    test_epsilon = 0.07
    delta = 0.05
    compute_random = False
    list_of_binning = ['algo']
    if experiment == "SYNTH":  # if we generate q ourselves
        print('You are running the synthetic experiment...')

        power_base = 10
        U = power_base**power_base
        m_per_splits = 1000
        init_e = 0.05
        init_b = 0.3
        splits = 2
        S = 5
        ratio = 1.2
        distribution_type = 'STAIRS'  # STAIRS
        list_of_espilon_q = [0, init_e, init_e*1.5, init_e*2]
        list_of_title_q = [TYPE+':q '+str(e) for e in list_of_espilon_q]

    else:  # if we take q as the generative models we have, we load the samples.
        print('You are running the generative model experiment...')
        power_base = 6
        U = power_base**power_base
        m_per_splits = 10000
        S = 2
        ratio = 3
        splits = 10

    print("for this round m is ", m_per_splits*splits)
    print("and U is ", U)

    store_results = {}
    store_results_ranking = {}
    for algo in list_of_binning:
        store_results_ranking[algo] = []
    metrics = ['S', 'test', 'binning', 'A', 'nll']
    for metric in metrics:
        store_results[metric] = {}
        for title in list_of_title_q:
            store_results[metric][title] = {}

    if experiment == "SYNTH":
        if distribution_type == 'UNIFORM':
            ground_truth_p = prob_array_to_dict(makeUniProbArr(U))

        elif distribution_type == 'STAIRS':
            # posU = math.factorial(power_base)/U
            posU = 0.9
            ground_truth_p = make_stair_prob(
                U, posU=posU, ratio=ratio,  S=S)

        else:
            raise NotImplemented
        list_of_samples, list_of_pmf_q = load_samples(
            list_of_espilon_q, init_b, ground_truth_p, splits, U, m_per_splits, S, ratio, TYPE)

    else:
        dict_of_samples, ground_truth_p = load_generative_model_samples(
            power_base, num_files=10)
        list_of_samples = [val for _, val in dict_of_samples.items()]
        list_of_title_q = [key for key, _ in dict_of_samples.items()]
    trials = splits
    perform_our_test(list_of_samples, list_of_title_q,
                     S, trials, store_results)
    compute_NLL(list_of_samples, list_of_title_q, store_results)

    prefix = create_prefix_from_list(
        {'exp': experiment+TYPE, 'U': U, 'm_per_splits': m_per_splits, 'splits': splits, 'S': S, 'ratio': ratio, 'b': init_b, 'e': init_e})
    rows = []
    for q_name in list_of_title_q:
        values = [np.mean(store_results['nll'][q_name])]
        for key, val in store_results['A'][q_name].items():
            values.append(np.mean(val))
        rows.append([q_name] + values)
    top = [''] + ['nll'] + ['$B_'+str(B)+'$' for B in store_results['A'][q_name].keys()]
    build_latex_table([top]+rows, caption=TYPE + ' m/Omega' +
                      str((m_per_splits*splits)/U) + ' S:'+str(S), label=prefix)
    store_for_plotting(
        data={'data': store_results['binning']}, title=prefix+'_binning')
    store_for_plotting(
        data={'data': store_results['test']}, title=prefix+'_hypothesis')
    store_for_plotting(
        data={'data': store_results['S']}, title=prefix+'_S')
    store_for_plotting(
        data={'data': store_results['nll']}, title=prefix+'_nll')
