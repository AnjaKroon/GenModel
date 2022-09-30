from binned import p_to_bp
from discrete import makeUniProbArr, errFunct, genValArr, prob_array_to_dict, prob_dict_to_array, sampleSpecificProbDist
from gen_S import empirical_dist, test_to_reject_chi_square
from plot_utils import plot_S_stat, put_on_plot
from sampling.poisson import poisson_empirical_dist
import sys
import numpy as np
import random


def get_chi_square(trials, U, m, tempered, B=2):

    uni_prob_arr = makeUniProbArr(U)
    prob_array = uni_prob_arr
    if tempered:
        prob_array = errFunct(U, uni_prob_arr, e, b)
    chi_result_trials = []

    prob_hist = p_to_bp(prob_array_to_dict(prob_array), U, B)
    prob_array = prob_dict_to_array(prob_hist, B)

    uni_prob_hist = p_to_bp(prob_array_to_dict(uni_prob_arr), U, B)
    uni_prob_array = prob_dict_to_array(uni_prob_hist, B)    
    U = B
    
    
    for i in range(trials):
        new_samples = sampleSpecificProbDist(genValArr(U), prob_array, m)

        p_emp = poisson_empirical_dist(
            U, m, new_samples, lambda m: sampleSpecificProbDist(genValArr(U), prob_array, m))
        p_emp_array = prob_dict_to_array(p_emp, U) 
        s = np.sum(p_emp_array)
        rejected = test_to_reject_chi_square(uni_prob_array, p_emp_array)
        chi_result_trials.append(rejected)
    return chi_result_trials


if __name__ == '__main__':
    # Set the random seed
    np.random.seed(2)
    random.seed(3)

    e = 0.4
    b = 50
    trials = 50

    chi_uni_binned = []
    chi_small_tempered_binned = []
    chi_big_tempered_binned = []
    rank_binned = []
    list_U = [ 250]
    list_M = [10000]
    # for m in list_M:
    for m in list_M:
        print("for this round m is ", m)
        for U in list_U:
            
            # uniform
            chi_binned_uni_U = get_chi_square(
                trials, U, m, tempered=False, B=5)

            # tempered
            chi_binned_tempered_U = get_chi_square(
                trials, U, m, tempered=True,  B=5)

            succeded_to_reject_uniform = np.mean(chi_binned_uni_U)
            succeded_to_reject_tempered = np.mean(chi_binned_tempered_U)
            chi_uni_binned.append(succeded_to_reject_uniform)
            chi_small_tempered_binned.append(succeded_to_reject_tempered)
