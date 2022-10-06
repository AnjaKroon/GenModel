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
    trials = 100

    list_U = [10**10]
    list_M = [1000]
    # for m in list_M:
    for m in list_M:
        print("for this round m is ", m)
        for U in list_U:
            print("and U is ",U)
            B_uni = []
            B_temper = []
            B_mid_temper = []
            B_easy_temper = []
            S_uni = []
            S_temper = []
            S_mid_temper = []
            S_easy_temper = []
            Bs = [2, 3, 4, 5, 6, 7, 8]

            ground_truth_p = prob_array_to_dict(makeUniProbArr(U))

            ground_truth_samples_list = generate_samples_scalable(
                trials, U, m, tempered=False, e=0, b=100)
            tempered_samples_list = generate_samples_scalable(
                trials, U, m, tempered=True, e=init_e, b=init_b)
            mid_tempered_samples_list = generate_samples_scalable(
                trials, U, m, tempered=True, e=init_e*1.5, b=init_b)
            easy_tempered_samples_list = generate_samples_scalable(
                trials, U, m, tempered=True, e=init_e*2, b=init_b)

            for B in Bs:

                def do_everything_here(all_samples_list, all_s_list,  trials, U, m, tempered, e, b, B):

                    S_trials_for_this_B_list = compute_stats(
                        all_samples_list, ground_truth_p,U,  B=B, stat_func=genSstat)
                    all_s_list.append(S_trials_for_this_B_list)

                do_everything_here(ground_truth_samples_list, S_uni, trials, U,
                                   m, tempered=False, e=0, b=100, B=B)
                do_everything_here(tempered_samples_list, S_temper, trials,
                                   U, m, tempered=True, e=init_e, b=init_b, B=B)
                do_everything_here(mid_tempered_samples_list, S_mid_temper, trials,
                                   U, m, tempered=True, e=init_e*1.5, b=init_b, B=B)
                do_everything_here(easy_tempered_samples_list, S_easy_temper, trials,
                                   U, m, tempered=True, e=init_e*2, b=init_b, B=B)

            # put_on_plot(Bs, {'uni': B_uni})
            # put_on_plot(Bs, {'hard tempered': B_temper})
            # put_on_plot(Bs, {'mid tempered': B_mid_temper})
            # put_on_plot(Bs, {'easy tempered': B_easy_temper})

            # plot_stat('chi_square_prob.pdf', 'Bins', 'goodness of fit')

            put_on_plot(Bs, {'uni': S_uni})
            put_on_plot(Bs, {'hard tempered': S_temper})
            put_on_plot(Bs, {'mid tempered': S_mid_temper})
            put_on_plot(Bs, {'easy tempered': S_easy_temper})

            plot_stat('S.pdf', 'Bins', 'S')
