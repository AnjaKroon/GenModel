
import random
import sys
import pandas as pd
import scipy
from tqdm import tqdm
from sampling.poisson import poisson_empirical_dist
from statistic.binned import p_to_bp_algo, p_to_bp_random, p_to_bp_with_index
from sampling.discrete import makeUniProbArr, errFunct, genValArr, prob_array_to_dict, prob_dict_to_array, sampleSpecificProbDist, scalabale_sample_distribution
import numpy as np

# compute kendall tau ranking score
# we have [ [all trials model 1], [all trials model 2], ...]
# the ground truth rank is model 1 < model 2 < ...


def get_ranking_results(all_models_list_stats):
    number_of_models = len(all_models_list_stats)
    number_of_trials = len(all_models_list_stats[0])
    ground_truth_ranking = list(range(number_of_models))
    kendalltau_ranking_metric_all_trials = []

    for i in range(number_of_trials):
        trial_i_all_models = [trials[i] for trials in all_models_list_stats]
        ranking_i = np.argsort(trial_i_all_models)
        kendalltau = scipy.stats.kendalltau(ranking_i, ground_truth_ranking)[
            0]  # only takes the statistic, higher is better
        kendalltau_ranking_metric_all_trials.append(kendalltau)
    return kendalltau_ranking_metric_all_trials


# default is without poissonization, we remove the option because it is not scalable
# this return a list of empirical distribution dict, one for each trial
# TODO large scale poisson.


def generate_samples_scalable(ground_truth_p, trials, U, m, tempered, e, b):
    all_trials_p_emp = []
    # first, check if the ground truth is given in the optimized format
    is_optimized = type(list(ground_truth_p.values())[0]) is dict
    if not is_optimized:  # the space is small enough to follow normal sampling procedure
        prob_array = prob_dict_to_array(ground_truth_p, U)
        if tempered:
            prob_array = errFunct(U, prob_array, e, b)
        for _ in range(trials):

            new_samples = sampleSpecificProbDist(
                genValArr(U), prob_array, m)

            p_emp_dict = empirical_dist_no_zero(m, new_samples)
            all_trials_p_emp.append(p_emp_dict)
    else:  # the space is too big
        prob_optimized_dict = ground_truth_p
        if tempered:
            prob_optimized_dict = errFunct(U, ground_truth_p, e, b)
        for _ in range(trials):

            new_samples = scalabale_sample_distribution(
                U, prob_optimized_dict, m)
            p_emp_dict = empirical_dist_no_zero(m, new_samples)
            all_trials_p_emp.append(p_emp_dict)
    return all_trials_p_emp


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


def test_for_l2(q_samples, m, alpha_signi, U):
    list_x_in_samples = []
    for x, num in enumerate(q_samples):
        list_x_in_samples = list_x_in_samples + [x for _ in range(num)]
    random.shuffle(list_x_in_samples)
    t = int(5 * np.log2(1/alpha_signi))
    sub_m = int(m/t)
    error = np.sqrt(10*np.sqrt(U)/(sub_m-1))
   
    all_col = []
    # count the number of pairwise collision in subset sample sub_m
    for index_subset in range(t):
        col = 0
        choose_2_from_m = 0
        subsamples = list_x_in_samples[sub_m *
                                       index_subset: sub_m*(index_subset+1)]
        for i in range(sub_m):
            for j in range(i+1, sub_m):
                choose_2_from_m += 1  # just do +1 cause I am lazy
                if subsamples[i] == subsamples[j]:
                    col += 1
            all_col.append(col/choose_2_from_m)
    return np.median(all_col), error


def reject_if_bad_test(prob_array, q_emp_array, m):
    # recover histrogram
    q_samples = [int(m*p) for p in q_emp_array]
    # first step, transform stair to uniform
    q2_estimate, error = test_for_l2(q_samples, m, 1/3, len(q_emp_array))
    p2 = np.sum([p**2 for p in prob_array])
    lower_bound_l2 = p2 + q2_estimate -2 * np.sqrt(q2_estimate) * max(prob_array)
    lower_bound_l2 = np.sqrt(lower_bound_l2)
    bigger_error_to_test =  lower_bound_l2 - error
    
    return bigger_error_to_test


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


def search_list_for_value(array, val):
    if val in array:
        index = array.index(val)
        return index
    return None


def chi_square_stat(uni_prob_array, p_emp_array):
    a = np.sum(uni_prob_array)
    b = np.sum(p_emp_array)
    key_zero = search_list_for_value(uni_prob_array, 0)
    if key_zero is not None and p_emp_array[key_zero] == 0:
        del uni_prob_array[key_zero]
        del p_emp_array[key_zero]
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
