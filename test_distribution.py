from binned import p_to_bp
from discrete import makeUniProbArr, errFunct, genValArr, prob_array_to_dict, prob_dict_to_array, sampleSpecificProbDist
from gen_S import test_to_reject_chi_square, chi_square_stat
import matplotlib.pyplot as plt
from plot_utils import plot_stat, put_on_plot
from sampling.poisson import poisson_empirical_dist

import numpy as np
import random


def get_chi_square(trials, U, m, tempered, e, b, B):

    uni_prob_arr = makeUniProbArr(U)
    prob_array = uni_prob_arr
    if tempered:
        prob_array = errFunct(U, uni_prob_arr, e, b)
    chi_result_trials = []

    prob_hist, _ = p_to_bp(prob_array_to_dict(prob_array), U, B)
    prob_array = prob_dict_to_array(prob_hist, B)
    
    uni_prob_hist, _ = p_to_bp(prob_array_to_dict(uni_prob_arr), U, B)
    uni_prob_array = prob_dict_to_array(uni_prob_hist, B)
    U = B

    for _ in range(trials):
        new_samples = sampleSpecificProbDist(genValArr(U), prob_array, m)

        p_emp = poisson_empirical_dist(
            U, m, new_samples, lambda m: sampleSpecificProbDist(genValArr(U), prob_array, m))
        p_emp_array = prob_dict_to_array(p_emp, U)
        shoud_be_one = np.sum(p_emp_array)
        stat = chi_square_stat(uni_prob_array, p_emp_array)
        chi_result_trials.append(stat)
    return chi_result_trials


if __name__ == '__main__':
    # Set the random seed
    np.random.seed(2)
    random.seed(4)

    init_e = 0.2
    init_b = 60
    trials = 100

    chi_uni_binned = []
    chi_small_tempered_binned = []
    chi_big_tempered_binned = []
    rank_binned = []
    list_U = [250]
    list_M = [10000]
    # for m in list_M:
    for m in list_M:
        print("for this round m is ", m)
        for U in list_U:
            B_uni = []
            B_temper = []
            B_mid_temper = []
            B_harder_temper = []
            Bs = [2, 3, 4, 5]
            for B in Bs:
                # uniform
                chi_binned_uni_U = get_chi_square(
                    trials, U, m, tempered=False, e=0, b=100, B=B)
                
                # tempered
                chi_binned_tempered_U = get_chi_square(
                    trials, U, m, tempered=True, e=init_e, b=init_b, B=B)
                
                # tempered
                chi_binned_mid_tempered_U = get_chi_square(
                    trials, U, m, tempered=True, e=init_e+0.05, b=init_b, B=B)
               
                # tempered
                chi_binned_harder_tempered_U = get_chi_square(
                    trials, U, m, tempered=True, e=init_e+0.1, b=init_b, B=B)
                

                B_uni.append(chi_binned_uni_U)
                B_temper.append(chi_binned_tempered_U)
                B_mid_temper.append(chi_binned_mid_tempered_U)
                B_harder_temper.append(chi_binned_harder_tempered_U)
            
            put_on_plot(Bs, {'uni':B_uni})
            put_on_plot(Bs, {'hard tempered':B_temper})
            put_on_plot(Bs, {'mid tempered':B_mid_temper})
            put_on_plot(Bs, {'easy tempered':B_harder_temper})

            plot_stat(None, None, 'chi_square_prob.pdf', 'Bins', 'goodness of fit')
            

