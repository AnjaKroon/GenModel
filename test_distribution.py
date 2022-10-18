from asyncio import start_unix_server
import math
import os
from tqdm import tqdm
from discrete import makeUniProbArr, prob_array_to_dict
from stair import make_stair_prob
from statistic.generate_statistics import get_ranking_results, perform_binning_and_compute_stats, genSstat, generate_samples_scalable
from plot_utils import plot_stat, put_on_plot
import numpy as np
import random


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

    Bs = [4, 5, 6, 7, 8]
    power_base = 6
    list_U = [power_base**power_base]
    list_M = [1000]

    list_of_binning_algo = ['algo', 'random', 'none']
    list_of_espilon_q = [0, init_e, init_e*1.5, init_e*2]
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

            # obtain the samples
            list_of_samples = []
            for e in list_of_espilon_q:
                if e == 0:
                    samples = generate_samples_scalable(
                        ground_truth_p, trials, U, m, tempered=False, e=0, b=100)
                else:
                    samples = generate_samples_scalable(
                        ground_truth_p, trials, U, m, tempered=True, e=e, b=init_b)
                list_of_samples.append(samples)

            for B in tqdm(Bs):  # For each bin granularity

                for i, all_samples_list in enumerate(list_of_samples):
                   
                    S_trials_for_this_B_list = perform_binning_and_compute_stats(
                        all_samples_list, ground_truth_p, U, B, stat_func=genSstat)

                    algo_stat = [i['B_algo'] for i in S_trials_for_this_B_list]
                    random_stat = [i['B_random']
                                   for i in S_trials_for_this_B_list]
                    
                    list_of_results_stats[0][i].append(algo_stat)
                    list_of_results_stats[1][i].append(random_stat)
                    #list_of_results_stats[2][i].append(no_binning)

            print('Generating S plots...')

            # plotting_dict_no_binning = {}
            # for i, title in enumerate(list_of_title_q):
            #     plotting_dict_no_binning[title] = list_of_results_stats[2][i]
            # put_on_plot(Bs, plotting_dict_no_binning)
            # prefix_title = 'U_' + str(U) + '_m_' + str(m)
            # prefix_title = os.path.join('figures', prefix_title)
            # plot_stat(prefix_title+'_nobinning_S.pdf', 'Bins',
            #           'Empirical Total Variation Error')

            plotting_dict_algo = {}
            for i, title in enumerate(list_of_title_q):
                plotting_dict_algo[title] = list_of_results_stats[0][i]
            put_on_plot(Bs, plotting_dict_algo)
            prefix_title = 'U_' + str(U) + '_m_' + str(m)
            prefix_title = os.path.join('figures', prefix_title)
            plot_stat(prefix_title+'_algo_S.pdf', 'Bins',
                      'Empirical Total Variation Error')

            plotting_dict_random = {}
            for i, title in enumerate(list_of_title_q):
                plotting_dict_random[title] = list_of_results_stats[1][i]
            put_on_plot(Bs, plotting_dict_random)

            # error of _____ w.r.t ground truth
            # for no temper, samples are generated from uniform dist
            # for heavily temper, samples are generated from heavily tempered distribution
            # thus, the gen model should have an easier time distinguishing the heavily
            # tempered case and will easily give it a lower rank

            plot_stat(prefix_title+'_random_S.pdf', 'Bins',
                      'Empirical total variation error')

            print('Generating ranking plots...')

            algo_ranking_results_all_trials_all_Bs = []
            for i in range(len(Bs)):
                list_at_B = [q[i] for q in list_of_results_stats[0]]
                algo_ranking_results_all_trials_all_Bs.append(
                    get_ranking_results(list_at_B))

            random_ranking_results_all_trials_all_Bs = []
            for i in range(len(Bs)):
                list_at_B = [q[i] for q in list_of_results_stats[1]]
                random_ranking_results_all_trials_all_Bs.append(
                    get_ranking_results(list_at_B))

            put_on_plot(Bs, {'algo': algo_ranking_results_all_trials_all_Bs,
                        'random': random_ranking_results_all_trials_all_Bs})
            plot_stat(prefix_title + '_ranking.pdf',
                      'Bins', 'Kendall tau distance')
