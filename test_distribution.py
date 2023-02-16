from audioop import lin2adpcm
import scipy
from sklearn.covariance import log_likelihood
from file_helper import create_prefix_from_list, load_samples, store_for_plotting
from sampling.loading_samples import load_generative_model_samples
from sampling.stair import make_stair_prob
from sampling.discrete import makeUniProbArr, prob_array_to_dict
from statistic.binning_algo import binning_on_samples
from statistic.generate_statistics import compute_norm, genSstat, get_pmf_val, get_ranking_results, reject_if_bad_test
import numpy as np
import random
import math
from tqdm import tqdm
from table_helper import build_latex_table


def compute_NLL(ground_truth_samples, list_of_pmf_q, list_of_title_q, store_results):
    all_log_likelihoods = []
    for i, q_name in enumerate(list_of_title_q):
        log_likelihoods = []
        pmf = list_of_pmf_q[i]
        q_name = list_of_title_q[i]

        for trial in ground_truth_samples:
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
        store_results['std_nll'][q_name] = np.std(log_likelihoods)
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


def perform_our_test(list_of_samples, list_of_title_q, S, trials, store_results, list_of_pmf_q=None):
    # step one consolidate all samples to one sample set
    consolidated_samples = []
    for all_samples_list in list_of_samples:
        consolidated_samples.append(consolidate(all_samples_list))

    Bs = list(range(S+1, 9))
    for B in tqdm(Bs):  # For each bin granularity

        for i, consolidated_samples_baseline in enumerate(consolidated_samples):
            pmf_q = None
            if list_of_pmf_q is not None:
                pmf_q = list_of_pmf_q[i]
            list_binned = binning_on_samples(
                consolidated_samples_baseline, trials, ground_truth_p, B, pmf_q)
            # run statistical test
            results = [reject_if_bad_test(
                trial['p'], trial['q'], splits*m_per_splits, epsilon=test_epsilon, delta=delta) for trial in list_binned]
            test = [i['close_enough'] for i in results]
            A = [i['emp_dtv'] for i in results]
            error = [i['e_test'] for i in results]

            q_name = list_of_title_q[i]
            if pmf_q is not None:
                true_norm_results = [compute_norm(
                    trial['p'], trial['q_true']) for trial in list_binned]
                l2 = [i['l2'] for i in true_norm_results]
                l1 = [i['l1'] for i in true_norm_results]
                store_results['l1'][q_name][B] = l1

            store_results['test'][q_name][B] = test
            store_results['A'][q_name][B] = A

            store_results['e'][q_name][B] = error
            store_results['binning'][q_name][B] = list_binned


def float_to_print(number, num_d=3):
    if num_d == 3:
        return '{:.3f}'.format(number)
    elif num_d ==2:
        return  '{:.2f}'.format(number)
    elif num_d ==1:
        return  '{:.1f}'.format(number)

if __name__ == '__main__':
    # Set the random seed
    np.random.seed(3)
    random.seed(3)
    experiment = "GEN"  # either SYNTH or GEN
    TYPE = "SHARP"  # TAIL, SHARP, FLAT
    test_epsilon = None
    delta = 0.5
    compute_random = False
    list_of_binning = ['algo']
    if experiment == "SYNTH":  # if we generate q ourselves
        print('You are running the synthetic experiment...')

        power_base = 10
        U = power_base**power_base
        m_per_splits = 20000
        init_e = 0.05
        init_b = 0.5
        splits = 10
        S = 4
        ratio = 5
        distribution_type = 'STAIRS'  # STAIRS
        list_of_espilon_q = [0, init_e, init_e*1.5, init_e*2]
        list_of_title_q = [TYPE+':q ' +
                           float_to_print(e) for e in list_of_espilon_q]

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
    metrics = ['S', 'test', 'binning', 'A', 'nll', 'e', 'std_nll', 'l1']
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
        store_results = {}
        store_results_ranking = {}
        for algo in list_of_binning:
            store_results_ranking[algo] = []

        for metric in metrics:
            store_results[metric] = {}
            for title in list_of_title_q:
                store_results[metric][title] = {}
    else:
        dict_of_samples, ground_truth_p = load_generative_model_samples(
            power_base, num_files=10)
        list_of_samples = [val for _, val in dict_of_samples.items()]
        list_of_title_q = [key for key, _ in dict_of_samples.items()]
        store_results = {}
        store_results_ranking = {}
        for algo in list_of_binning:
            store_results_ranking[algo] = []
        for metric in metrics:
            store_results[metric] = {}
            for title in list_of_title_q:
                store_results[metric][title] = {}
        list_of_pmf_q = None
    trials = 50
    perform_our_test(list_of_samples, list_of_title_q,
                     S, trials, store_results, list_of_pmf_q)
    ground_truth_samples = list_of_samples[0]
    if list_of_pmf_q is not None:
        compute_NLL(ground_truth_samples, list_of_pmf_q,
                    list_of_title_q, store_results)

    #coverage_baselines(ground_truth_samples, list_of_samples)
    if experiment == "SYNTH":
        prefix = create_prefix_from_list(
            {'exp': experiment+TYPE, 'U': U, 'm_per_splits': m_per_splits, 'splits': splits, 'S': S, 'ratio': ratio, 'b': init_b, 'e': init_e})
    else:
        prefix = create_prefix_from_list(
            {'exp': experiment+TYPE, 'U': U, 'm_per_splits': m_per_splits, 'splits': splits, 'S': S, 'ratio': ratio})
    store_for_plotting(data={'data': store_results}, title=prefix)

    rows = []
    for q_name in list_of_title_q:
        values = []
        if list_of_pmf_q is not None:
            # values = [float_to_print(np.mean(store_results['nll'][q_name])) +
            #           '$\pm$' + float_to_print(store_results['std_nll'][q_name])]
            values = [float_to_print(np.mean(store_results['nll'][q_name])) ]
        
        for key, val in store_results['A'][q_name].items():
            std = np.mean((store_results['e'][q_name][key]))
            #std = np.std(val)

            #values.append(float_to_print(np.mean(val)) +
            #               '$\pm$' + float_to_print(std))
            values.append(float_to_print(np.mean(val),num_d=3) )
            # values.append(float_to_print(np.mean(store_results['l1'][q_name][key])))
        rows.append([q_name] + values)
    top = ['']
    if list_of_pmf_q is not None:
        top = top + ['nll']
    for B in store_results['A'][q_name].keys():
        top = top + ['$B_'+str(B)+'$']
        #top = top + [ '$tv$']
    build_latex_table([top]+rows, caption=TYPE + ' m/Omega' +
                      float_to_print((m_per_splits*splits)/U) + ' S:'+str(S), label=prefix)
