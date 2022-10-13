import itertools
from random import uniform
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete
import math
from time import time
from itertools import combinations, permutations
from read_pickle import read_pickle_file
from statistic.generate_statistics import genSstat, perform_binning_and_compute_stats
from binned import p_to_bp_random, transform_samples 

# OBSOLETE I THINK
def stair_mapping(incoming_X_arr):
    # takes an incoming arr of x's
    # R.V. X = x where x = [1,2,3,4,5,6]
    # if an x has x[0] > x[5] -- pr = 3/(2**6)
    # if an x has x[5] > x[0] -- pr = 1/(2**6)
    # if an x does not contain a full permutation, pr = 0

    permutations_6 = [1, 2, 3, 4, 5, 6]
    histo = {}
    for j in range(len(incoming_X_arr)):  # for each possible event in incoming X arr
        check_one_of_ea = [0, 0, 0, 0, 0, 0]
        for i in incoming_X_arr[j]:  # for each list item in each possible event
            if i == 1:
                check_one_of_ea[0] = 1
            elif i == 2:
                check_one_of_ea[1] = 1
            elif i == 3:
                check_one_of_ea[2] = 1
            elif i == 4:
                check_one_of_ea[3] = 1
            elif i == 5:
                check_one_of_ea[4] = 1
            elif i == 6:
                check_one_of_ea[5] = 1
        if (0 in check_one_of_ea):
            histo.update({str(incoming_X_arr[j]): 0})
        elif (incoming_X_arr[j][0] > incoming_X_arr[j][5]):
            histo.update({str(incoming_X_arr[j]): (3/(2**6))})
        elif (incoming_X_arr[j][0] < incoming_X_arr[j][5]):
            histo.update({str(incoming_X_arr[j]): (1/(2**6))})
        else:
            # should not trigger this but just to be safe
            histo.update({str(incoming_X_arr[j]): 0})
    # IMPORTANT: TO ADD TO A DICTIONARY, I HAD TO TURN THE x EVENT INTO A STRING. IF YOU WISH TO USE AS AN ARRAY NEED TO CONVERT BACK
    return histo


def make_stair_prob(U, posU, ratio, S):
    # From my understanding
    # Take U * posU to get the amount of U that will have a stair function
    # S is the total amount of steps thus (U*posU)/S is the amount of U that each step will take
    # ratio is highest pmf/lowest pmf -- representative of the amount of "y step" in between each stair
    # pmf of each value for the whole U will have to sum to 1
    # highest pmf / lowest pmf = ratio -- (?/common denom) / (?/common denom)
    U_with_stair = int(posU * U)

    # be careful to consider the case this may be fractional
    U_for_each_S = math.floor(U_with_stair/S)
    U_for_last_S = U_with_stair - (S-1)*U_for_each_S
    U_per_stairs = [U_for_each_S for i in range(S-1)]
    U_per_stairs.append(U_for_last_S)

    ratio_all_steps = list(np.arange(1, ratio, (ratio-1)/(S-1)))
    ratio_all_steps.append(ratio)
    p_first_floor = 1 / \
        (np.sum([U_per_stairs[i]*ratio_step for i,
         ratio_step in enumerate(ratio_all_steps)]))
    p_each_stair = [p_first_floor *
                    ratio_stair for ratio_stair in ratio_all_steps]
    verify_that_is_one = np.sum(
        [p_each_stair[i]*U_per_stairs[i] for i in range(S)])

    U_per_stairs.reverse()
    p_each_stair.reverse()
    current_dist = 0
    current_step = 0
    stair_histo = {}
    if U <= 7**7:
        for i, size_stair in enumerate(U_per_stairs):
            current_dist = p_each_stair[i]
            for _ in range(size_stair):
                current_dist = p_each_stair[i]
                stair_histo[current_step] = current_dist
                current_step += 1

    else:
        start_interval = 0
        for i, size_stair in enumerate(U_per_stairs):
            current_dist = p_each_stair[i]
            interval = [start_interval, start_interval+size_stair]
            stair_histo[i] = {'interval': interval, 'p': current_dist}
            start_interval += size_stair
    return stair_histo

# takes the samples


def samples_to_histo(samples):
    # TODO actually build the histo from the samples
    # get total size of the samples -- that will determine 'what to divide by'
    # determine the "probability to add to each item if present"
    amount_samples = len(samples)
    increment = 1/amount_samples
    empirical_dict = {}
    for item in samples:
        # get value currently and add the "increment"
        if str(item) in empirical_dict:
            # gets the value corresponding to the key (item, here)
            val = empirical_dict.get(str(item))
            val += increment
            empirical_dict.update({str(item): val})
        else:
            # add item to empirical_dict with the base "increment"
            empirical_dict.update({str(item): increment})
    
    # empirical_dict = {'1-2-3-4-5-6':0.8, '2-3-2-3-5-6':0.1,'6-5-2-3-4-1':0.1 }
    should_be_one = np.sum(list(empirical_dict.values()))
    print('should_be_one', should_be_one)
    return empirical_dict


# by default, it is 6
def build_ground_truth_dict():
    # return a dict with all permutation as keys, and the value are the ground truth pmf either (3/(2**6)) or (1/(2**6))
    # create 2D array with all permutations as keys
    # 6*5*4*3*2*1 = 720
    numbers = [1, 2, 3, 4, 5, 6]
    c = list(permutations(range(6), 6))
    # decided to make this a np array to match file type coming in from pickle.py
    c = np.array(c)
    # print(c)
    ground_truth_dict = {}
    for item in c:
        if item[0] < item[5]:
            ground_truth_dict.update(
                {str(item): round((3/(2*(6*5*4*3*2*1))), 5)})
        elif item[0] > item[5]:
            ground_truth_dict.update(
                {str(item): round((1/(2*(6*5*4*3*2*1))), 5)})
        else:
            ground_truth_dict.update({str(item): 0})
   
    should_be_one = np.sum(list(ground_truth_dict.values()))
    print('should_be_one', should_be_one)
    return ground_truth_dict


def get_type(comb):
    choices = {}
    for x in comb:
        if x not in choices:
            choices[x] = 1
        else:
            return 'not a permutation'
    if comb[0] < comb[5]:
        return 'likely'
    else:
        return 'rare'


def get_converting_dict():
    converting_dict = {}
    likely = []
    rare = []
    null_space = []

    for comb in itertools.product(range(6), repeat=6):
        comb_type = get_type(comb)
        if comb_type == 'likely':
            likely.append(comb)
        elif comb_type == 'rare':
            rare.append(comb)
        else:
            null_space.append(comb)
    index = 0
    for comb in likely:
        comb = str(np.array(comb))
        converting_dict[comb] = index
        index += 1
    for comb in rare:
        comb = str(np.array(comb))
        converting_dict[comb] = index
        index += 1
    for comb in null_space:
        comb = str(np.array(comb))
        converting_dict[comb] = index
        index += 1

    return converting_dict


def convert_key_sequence_to_int(histo_dict, KEY_CONVERTING_DICT):

    converted_dict = {}
    for key, val in histo_dict.items():
        if key  in KEY_CONVERTING_DICT:
            converted_dict[KEY_CONVERTING_DICT[key]] = val
    return converted_dict


if __name__ == '__main__':
    U = 6**6
    B = 4
    posU = 23  # pos U is % of U with > 0 pmf
    ratio = 3  # highest pmf/lowest pmf
    S = 6

    soln = {}
    X = [[1, 2, 3, 4, 5, 6], [3, 4, 5, 6, 1, 2], [1, 2, 2, 3, 4, 5],
         [4, 5, 6, 1, 2, 3], [1, 2, 4, 3, 5, 6], [6, 5, 4, 3, 2, 1]]
    # soln = stair_mapping(X)
    # print(soln)

    samples_from_file = read_pickle_file('100sample.pk')
    empirical_dict = samples_to_histo(samples_from_file)
    ground_truth_dict = build_ground_truth_dict()

    KEY_CONVERTING_DICT = get_converting_dict()
    empirical_dict = convert_key_sequence_to_int(
        empirical_dict, KEY_CONVERTING_DICT)
    ground_truth_dict = convert_key_sequence_to_int(
        ground_truth_dict, KEY_CONVERTING_DICT)
    
    print(empirical_dict)
    # print(ground_truth_dict)
    
    # BINNING
    '''
    
    print(U)
    B = 3 # later to be changed to an array rotating through
    b_p = p_to_bp_random(empirical_dict, U, B)   # transforms the probability distribution based on binning   
    b_out = transform_samples(b_p, sample_histo, p_samples, U, B)  # returns new samples based on binning
    # with new samples, would need to call samples_to_histo again to get a new dictionary corresponding to the new binned samples
    empirical_dict = samples_to_histo(b_out)
    # then get_converting_dict I think --  not sure about this one
    empirical_dict = convert_key_sequence_to_int(
        empirical_dict, KEY_CONVERTING_DICT)
    # then convert_key_sequence_to_int I think
    '''
    U = len(samples_from_file) # -- check
    print(perform_binning_and_compute_stats(empirical_dict, ground_truth_dict, U, B, stat_func=genSstat))



    # when we plot the dict in order, it should look like a stair
    x = list(ground_truth_dict.keys())
    x_sort_arg = np.argsort(x)
    y = list(ground_truth_dict.values())
    x = [x[i] for i in x_sort_arg]
    y = [y[i] for i in x_sort_arg]
    plt.plot(x, y)
    plt.title('This should look like a stair function once you are done')
    plt.show()
    plt.close()
    
    x = list(empirical_dict.keys())
    x_sort_arg = np.argsort(x)
    y = list(empirical_dict.values())
    x = [x[i] for i in x_sort_arg]
    y = [y[i] for i in x_sort_arg]
    plt.plot(x[0:720], y[0:720])
    plt.title('This should approach the stair function, maybe it is failing.')
    plt.show()
    plt.close()

    # this should work
    perform_binning_and_compute_stats(
        [empirical_dict], ground_truth_dict, U, B, stat_func=genSstat)

    # U posU ratio and S are parameters that will define the stair function
    # stair_histo = make_stair_prob(U, posU, ratio, S)
    # print(np.sum(list(stair_histo.values())))
    # print(stair_histo)

