# objective of this code is to generate the S statistic according to algorithm 1
import numpy as np
import sys
import pandas as pd
import scipy
from sampling.poisson import poisson_empirical_dist
from binned import p_to_bp_random, p_to_bp_with_index
from discrete import makeUniProbArr, errFunct, genValArr, prob_array_to_dict, prob_dict_to_array, sampleSpecificProbDist, scalabale_sample_distribution

# default is without poissonization, we remove the option because it is not scalable
# this return a list of empirical distribution dict, one for each trial
# TODO large scale poisson.

def generate_samples_scalable(trials, U, m, tempered, e, b):
    all_trials_p_emp = []
    uni_prob_arr = makeUniProbArr(U)
    prob_array = uni_prob_arr
    if tempered:
        prob_array = errFunct(U, uni_prob_arr, e, b)
    for _ in range(trials):
        if U >7**7:
            new_samples = scalabale_sample_distribution(
                U, prob_array, m, flatten_dist=None)
        else:
            new_samples = sampleSpecificProbDist(genValArr(U), prob_array, m)
        p_emp_dict = empirical_dist_no_zero(m, new_samples)
        all_trials_p_emp.append(p_emp_dict)
    return all_trials_p_emp


# for this, we already have the samples, but they are not binned
def compute_stats(all_trials_p_emp, ground_truth_dict, U, B, stat_func):
    list_stat = []
    for p_emp_dict in all_trials_p_emp:
        binnned_p_hist, mapping_from_index_to_bin = p_to_bp_random(
            ground_truth_dict, U, B)
        binnned_q_hist = p_to_bp_with_index(
            p_emp_dict, U, B, mapping_from_index_to_bin)
        binnned_p_array = prob_dict_to_array(binnned_p_hist, B)
        binnned_q_array = prob_dict_to_array(binnned_q_hist, B)

        list_stat.append(stat_func(binnned_p_array, binnned_q_array))
    return list_stat


def generate_samples_and_compute_stat(trials, U, m, tempered, e, b, B, stat_func, with_poisson=True):
    uni_prob_arr = makeUniProbArr(U)
    prob_array = uni_prob_arr
    if tempered:
        prob_array = errFunct(U, uni_prob_arr, e, b)

    result_trials = []

    uni_prob_hist, mapping_from_index_to_bin = p_to_bp_random(
        prob_array_to_dict(uni_prob_arr), U, B)
    uni_prob_array = prob_dict_to_array(uni_prob_hist, B)

    prob_hist = p_to_bp_with_index(prob_array_to_dict(
        prob_array), U, B, mapping_from_index_to_bin)
    prob_array = prob_dict_to_array(prob_hist, B)

    U = B

    for _ in range(trials):
        new_samples = sampleSpecificProbDist(genValArr(U), prob_array, m)
        if with_poisson:
            p_emp = poisson_empirical_dist(
                U, m, new_samples, lambda m: sampleSpecificProbDist(genValArr(U), prob_array, m))

        else:
            p_emp = empirical_dist(
                U, m, sampleSpecificProbDist(genValArr(U), prob_array, m))
        p_emp_array = prob_dict_to_array(p_emp, U)
        shoud_be_one = np.sum(p_emp_array)
        stat = stat_func(uni_prob_array, p_emp_array)
        result_trials.append(stat)
    return result_trials


def get_chi_square(trials, U, m, tempered, e, b, B):
    result_trials = generate_samples_and_compute_stat(
        trials, U, m, tempered, e, b, B, chi_square_stat)
    return result_trials


def get_S(trials, U, m, tempered, e, b, B, with_poisson):

    result_trials = generate_samples_and_compute_stat(
        trials, U, m, tempered, e, b, B, genSstat, with_poisson)

    return result_trials


def empirical_dist(incoming_U, incoming_m, incoming_arr_samples):
    # Create a dictionary with size U. Note index will be shifted by 1 as i starts at 0 and numbers start at 1
    key = []
    for i in range(incoming_m):
        key = np.append(key, incoming_arr_samples[i])
    value = 0
    histo = dict.fromkeys(key, int(value))
    # for each value in the samples array, add +1 to the frequency that the value corresponds to in histo
    for i in range(len(incoming_arr_samples)):
        val = histo.get(incoming_arr_samples[i])
        histo.update({incoming_arr_samples[i]: (val+1)})
    # Adding the zeros into the histogram
    # FLO: this is not necessary, the best is to have the non existing entries as begin 0 by default (to save memory)
    histo_with_zeros = {}
    for i in range(incoming_U):

        # basically, if it is not in the samples, you want to assign it to a value of zero with a key corresponding to i
        if not isinstance(histo.get(i), int):
            histo_with_zeros.update({i: 0})
        else:  # else you would add the histo type and value to the dictionary
            histo_with_zeros.update({i: histo.get(i)/incoming_m})
    p_emp = histo_with_zeros
    return p_emp


def empirical_dist_no_zero(incoming_m, incoming_array_samples):
    p_emp = {}
    for sample in incoming_array_samples:
        if sample in p_emp:
            p_emp[sample] += 1/incoming_m
        else:
            p_emp[sample] = 1/incoming_m

    return p_emp


def intoCSV(arr, U, m, e, b):
    two_dim_arr = np.array(list(arr.items()))
    DF = pd.DataFrame(two_dim_arr)
    e = int(e*100)
    DF.to_csv(f'histo_{U}_{m}_{e}_{b}.csv', index=False, header=False)
    return


def test_to_reject_chi_square(uni_prob_array, p_emp_array):
    a = np.sum(uni_prob_array)
    b = np.sum(p_emp_array)
    chi_square_out = scipy.stats.chisquare(uni_prob_array, p_emp_array)
    p_value = chi_square_out[1]
    if p_value < 0.99:
        reject = True
    else:
        reject = False
    return reject


def chi_square_stat(uni_prob_array, p_emp_array):
    a = np.sum(uni_prob_array)
    b = np.sum(p_emp_array)
    chi_square_out = scipy.stats.chisquare(uni_prob_array, p_emp_array)
    p_value = chi_square_out[1]
    return p_value


def genSstat(uni_prob_array, p_emp_array):
    sum = 0
    for i, p_val_ground_truth in enumerate(uni_prob_array):
        emp_p_val = p_emp_array[i]
        sum += np.abs(emp_p_val-p_val_ground_truth)
    sum = sum/2
    return sum


if __name__ == '__main__':
    testCase = 1  # should be 1 or 2 depending on whether you want to run the program standalone or with a .sh script
    if testCase == 1:
        incoming_arr_samples = np.load('Gen_Samples.npy')
        if len(sys.argv) != 5:
            print("Usage:", sys.argv[0], "U m e b")
            sys.exit()
        U = int(sys.argv[1])
        m = int(sys.argv[2])
        # recall this value has been multiplied by 100 in sh script
        e = float(sys.argv[3])/100
        b = int(sys.argv[4])

    if testCase == 2:
        incoming_arr_samples = [2, 3, 3, 3, 4,
                                4, 5, 6, 7, 8, 9, 10, 11, 1, 14, 17]
        U = 19
        m = 16
        e = 0.1
        b = 100

    p_emp_dependent = empirical_dist(U, m, incoming_arr_samples)
    intoCSV(p_emp_dependent, U, m, e, b)  # Turning into .csv file
    s_statistic = genSstat(p_emp_dependent, U)
    print(U, " ",  m, " ", e, " ", b, " ", s_statistic)

    # TODO: also need to remove dependencies to make the histogram independent? "sample the samples"? need to follow up on the procedure for that
