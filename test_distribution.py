from tqdm import tqdm
from binned import p_to_bp_random
from discrete import makeUniProbArr, errFunct, genValArr, prob_array_to_dict, prob_dict_to_array, sampleSpecificProbDist
from statistic.generate_statistics import chi_square_stat, compute_stats, genSstat, generate_samples_scalable, get_S, get_chi_square
import matplotlib.pyplot as plt
from plot_utils import plot_stat, put_on_plot
from sampling.poisson import poisson_empirical_dist

import numpy as np
import random


if __name__ == '__main__':
    # Set the random seed
    np.random.seed(3)
    random.seed(3)

    init_e = 0.2
    init_b = 60
    trials = 50

    Bs = [2, 3, 4, 5, 6, 7, 8]
    list_U = [6**6]
    list_M = [10000]
    # for m in list_M:
    for m in list_M:
        print("for this round m is ", m)
        for U in list_U:
            print("and U is ", U)

            stat_uni = []
            stat_temper = []
            stat_mid_temper = []
            stat_easy_temper = []

            ground_truth_p = prob_array_to_dict(makeUniProbArr(U))
            # first, we generate all the samples here. The same samples should be reused for each B.
            # If it takes too much memory, we can put this in a for loop.
            ground_truth_samples_list = generate_samples_scalable(
                trials, U, m, tempered=False, e=0, b=100)
            tempered_samples_list = generate_samples_scalable(
                trials, U, m, tempered=True, e=init_e, b=init_b)
            mid_tempered_samples_list = generate_samples_scalable(
                trials, U, m, tempered=True, e=init_e*1.5, b=init_b)
            easy_tempered_samples_list = generate_samples_scalable(
                trials, U, m, tempered=True, e=init_e*2, b=init_b)

            for B in tqdm(Bs):  # For each bin granularity
                
                # this fonction takes the samples, compile statistics, then store the result in all_stat_list
                def compile_all_stats(all_samples_list, all_stat_list,   U,  B): 

                    S_trials_for_this_B_list = compute_stats(
                        all_samples_list, ground_truth_p, U, B, stat_func=genSstat)
                    all_stat_list.append(S_trials_for_this_B_list)

                compile_all_stats(ground_truth_samples_list, stat_uni, U,B=B) # compile stats for ground truth
                compile_all_stats(tempered_samples_list, stat_temper,U,B=B) # compile stats for tempered
                compile_all_stats(mid_tempered_samples_list,stat_mid_temper, U,B=B) # compile stats for tempered
                compile_all_stats(easy_tempered_samples_list,stat_easy_temper, U, B=B) # compile stats for tempered

            # put_on_plot(Bs, {'uni': B_uni})
            # put_on_plot(Bs, {'hard tempered': B_temper})
            # put_on_plot(Bs, {'mid tempered': B_mid_temper})
            # put_on_plot(Bs, {'easy tempered': B_easy_temper})

            # plot_stat('chi_square_prob.pdf', 'Bins', 'goodness of fit')
            print('Generating S plots...')
            put_on_plot(Bs, {'uni': stat_uni})
            put_on_plot(Bs, {'hard tempered': stat_temper})
            put_on_plot(Bs, {'mid tempered': stat_mid_temper})
            put_on_plot(Bs, {'easy tempered': stat_easy_temper})

            plot_stat('S.pdf', 'Bins', 'S')
