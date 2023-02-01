# The objective of this code is to create samples from a slightly skewed uniform probability distribution for discrete events.

from math import remainder
import sympy
import random
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete
from tqdm import tqdm

# transform a prob array into a dictionnary


def prob_array_to_dict(prob_array):
    if callable(prob_array):  # the array is actually a function, we return it as is
        return prob_array
    prob_hist = {}
    U = len(prob_array)
    for i in range(U):
        prob_hist[i] = prob_array[i]
    return prob_hist
# transform a prob dictionnary into an array


def prob_dict_to_array(prob_hist, U):
    prob_array = []
    all_keys = list(prob_hist.keys())
    all_keys.sort()
    for key in range(U):
        if key in prob_hist:
            prob_array.append(prob_hist[key])
        else:
            prob_array.append(0)
    return prob_array

# Given: U, probability space
# Returns: np array with a uniform distribution of probability space |U|


def makeUniformDie(U):
    # I think this part may take a lot of time, perhaps think of a faster way
    sides_of_die = [None]*U
    for i in range(U):
        sides_of_die[i] = i
    return sides_of_die

# Given: ArrayToSample is the array you would like to sample, m is the amount of times you would like to sample
# Returns: a numpy array with the samples in it


def sampleAnArray(ArrayToSample, m):
    samples = np.random.choice(ArrayToSample, size=m)
    return samples

# Given: U is size of prob space, array is a np array representing probability distribution for each item in probability space,
# xAxis label, yAxis label, and title label
# Returns: a bar graph with each bar representing the probability of that x value to be chosen


def plotProbDist(U, array, xAxis, yAxis, title):
    # a bar graph with U on x axis and array values matching the y axis
    x_ax = np.arange(start=1, stop=U+1, step=1)
    plt.bar(x_ax, array, width=1)
    #count, bins, ignored = plt.hist(array, U, density=True)
    plt.axhline(y=float(1/U), color='r', linestyle='-')
    plt.xlabel(xAxis)
    plt.ylabel(yAxis)
    plt.title(title)
    plt.savefig("ModProbDist.png")


# Given: U the size of probability space
# Returns: the uniform discrete probability distribution
def makeUniProbArr(U):
    prob = 1/U
    if U > 10e5:  # at this scale, we dont return am array but a function
        def get_prob(x):
            return 1/U
        return get_prob
    prob_arr = []
    for i in range(U):
        prob_arr.append(prob)
    return prob_arr

# basic binary search


def find_in_sorted_intervals(query, sorted_intervals):
    #current_list_intervals = sorted_intervals
    current_list_index = range(len(sorted_intervals))
    while True:
        # check if we have tiny list
        if len(current_list_index) == 0:
            return -1
        else:
            half_lookup = int(len(current_list_index)/2)
            current_index = current_list_index[half_lookup]
            current_interval = sorted_intervals[current_index]

            if current_interval[0] <= query and current_interval[1] > query:
                return current_index
            elif current_interval[1] <= query:
                current_list_index = current_list_index[half_lookup+1:]

            else:
                current_list_index = current_list_index[:half_lookup]


def find_interval(query, intervals):

    index_sorted = find_in_sorted_intervals(query, intervals)
    return index_sorted

# Given: U as size of probability space, array as the probability distribution (assumed uniform coming in)
# e which is the total amount of error that is introduced in the probability distribution array, and
# percent_to_modify
# Returns: new probability distribution.
# percent_to_modify_null is the percentage of error to be situaded in zero space


def get_overap(interval_M, interval_remain):  # return remainder from 2
    start_M = interval_M[0]
    end_M = interval_M[1]
    start_R = interval_remain[0]
    end_R = interval_remain[1]
    if end_R <= start_M or end_M <= start_R:  # no overlap
        return None, [], []
    elif start_M <= start_R:  # [0,..] [5,...]

        new_interval = (start_R, end_M)
        if end_M < end_R:
            post_remain = (end_M, end_R)
            return new_interval, [], [post_remain]
        else:
            return new_interval, [], []
    else:  # start_M > start_R
        new_interval = (start_M, end_R)
        pre_remain = (start_R, start_M)
        if end_M < end_R:
            post_remain = (end_M, end_R)
            return new_interval, [pre_remain],  [post_remain]
        else:
            return new_interval, [pre_remain], []


def errFunct(U, init_array, e, percent_to_modify, percent_to_modify_null, TYPE):

    is_optimized = not type(init_array) is list
    percent_pos_space = percent_to_modify-percent_to_modify_null
    assert percent_pos_space >= 0
    if not is_optimized:
        array = np.copy(init_array)  # copy to avoid modifying the passed array

        U_pos = np.where(array)[0].shape[0]  # count the positive space
    else:
        prob_optimized_dict = init_array
        mass_in_each_part = [(val['interval'][1] - val['interval'][0]) * val['p']
                             for key, val in prob_optimized_dict.items()]
        should_be_one = np.sum(mass_in_each_part)
        size_each_regions = [(val['interval'][1] - val['interval'][0])
                             for _, val in prob_optimized_dict.items()]
        intervals = [val['interval'] for _, val in prob_optimized_dict.items()]

        U_pos = np.sum(size_each_regions)  # count the positive space
    # works to modify probability dist. array, works for odd U
    # Tells us how many bins in the probability distribution we are changing
    amt_to_modify_pos = U_pos*(percent_pos_space)
    # on the null space, we can only add error.
    amt_to_modify_null = (U-U_pos) * (percent_to_modify_null)

    # If |U_pos| is odd, due to truncation in division, the 'extra bin' will go on the subtraction half.
    half_point = amt_to_modify_pos//2
    # That means for this case, the bins_last will need to 'redistribute' how much is subtracted per bin

    bins_added_in_pos = int(half_point)
    bins_removed = int(amt_to_modify_pos - half_point)
    bins_added_in_null = int(amt_to_modify_null)
    bins_added = bins_added_in_null + bins_added_in_pos
    e_per_section = e/2

    e_added = e_per_section/bins_added  # error amount to add per element
    e_removed = e_per_section/bins_removed  # error amount to subtract per element
    print('Total tv error should be ', e, ' it is ',
          e_added*bins_added+e_removed * bins_removed)
    print('Total l2^2 error is ', (e_added**2) *
          bins_added+(e_removed**2) * bins_removed)
    print('Total l2 error is ', np.sqrt((e_added**2)
          * bins_added+(e_removed**2) * bins_removed))
    if not is_optimized:

        """
        modification in the positive space
        """
        # randomly select where we add and remove.
        # We create a list with all indices of the pos. space, then shuffle the list.
        shuffled_indices_pos = list(range(U_pos))
        random.shuffle(shuffled_indices_pos)

        for i in shuffled_indices_pos[:bins_added_in_pos]:
            # adds same amount to first half of bins you wish to change
            array[i] = array[i] + e_added

        for i in shuffled_indices_pos[bins_added_in_pos:bins_added_in_pos+bins_removed]:
            # check that we are not removing too much

            if array[i] < e_removed:
                print('The negative error is too much', e_removed,
                      ', too concentrated', percent_pos_space)
                raise Exception
            # subtracts same amount to second half of bins you wish to change
            array[i] = array[i] - e_removed

        """
        modification in the null/zero space
        """
        # We just add in order in the null space
        for i in range(U_pos, U_pos+bins_added_in_null):
            # the array should be sorted s.t. the zero should be there
            assert array[i] == 0
            # adds same amount to first half of bins you wish to change
            array[i] = e_added

        # this check that the array sum up to one, to be a valid prob. I use assert close because something it won't be excatly one because of numerical error.
        should_be_one = np.sum(array)
        np.testing.assert_allclose(should_be_one, 1)
        return array
    else:

        """
        modification in the positive space
        """

        print('Starting the tempering process... Less randomized but way faster')
        new_inverse_tempered_dict = {}

        if TYPE == 'SHARP' or TYPE == 'FLAT':
            untouched_intervals = list(range(len(intervals)))
            # positive tempering
            num_per_interval = int((len(intervals))/2)
            if TYPE == 'SHARP':
                num_to_modify_pre_interval = int(
                    bins_added_in_pos/num_per_interval)
                num_to_modify_post_interval = int(
                    bins_removed/num_per_interval)
            else:
                num_to_modify_pre_interval = int(
                    bins_removed/num_per_interval)
                num_to_modify_post_interval = int(
                    bins_added_in_pos/num_per_interval)

            for i in range(num_per_interval):
                untouched_intervals.remove(i)
                interval = intervals[i]
                interval_to_modify = (
                    interval[0], interval[0]+num_to_modify_pre_interval)
                new_interval, _, interval_remains_post = get_overap(
                    interval_to_modify, interval_remain=interval)
                p_value_of_interval = prob_optimized_dict[i]['p']
                if TYPE == 'SHARP':
                    new_p_value = p_value_of_interval + e_added
                else:
                    new_p_value = p_value_of_interval - e_removed
                new_inverse_tempered_dict[new_p_value] = [new_interval]
                new_inverse_tempered_dict[p_value_of_interval] = interval_remains_post

            for i in range(num_per_interval):
                reverse_index = len(intervals)-1-i
                untouched_intervals.remove(reverse_index)
                interval = intervals[reverse_index]
                interval_to_modify = (
                    interval[0], interval[0]+num_to_modify_post_interval)
                new_interval, _, interval_remains_post = get_overap(
                    interval_to_modify, interval_remain=interval)
                p_value_of_interval = prob_optimized_dict[reverse_index]['p']
                if TYPE == 'SHARP':
                    new_p_value = p_value_of_interval - e_removed
                else:
                    new_p_value = p_value_of_interval + e_added
                new_inverse_tempered_dict[new_p_value] = [new_interval]
                new_inverse_tempered_dict[p_value_of_interval] = interval_remains_post
            for i in untouched_intervals:
                p_value_of_interval = prob_optimized_dict[i]['p']
                interval = intervals[i]
                new_inverse_tempered_dict[p_value_of_interval] = [interval]
        if TYPE == 'ANOM' or TYPE=='UNI' or TYPE=='TAIL':
            # for each interval, divide it in two and add to the start, remove from lasts.
            if TYPE =='TAIL':
                intervals = intervals[1:]
            e_added_pos = e_added
            if TYPE=='ANOM':
                bins_added = int(bins_added/4)
                e_in_zero = bins_added_in_null *e_added
                e_added_pos = (e_per_section-e_in_zero)/bins_added 
               
                

            num_to_add_per_interval = bins_added/len(intervals)
            remainder_add = bins_added % len(intervals)
            num_to_remove_per_interval = bins_removed/len(intervals)
            
            remainder_remove = bins_removed % len(intervals)

            num_to_add_per_interval = int(num_to_add_per_interval)
            num_to_remove_per_interval = int(num_to_remove_per_interval)
            for i, interval in enumerate(intervals):
                if i == len(intervals)-1: # last one, we add the remainder here
                    num_to_add_per_interval = num_to_add_per_interval+ remainder_add
                    num_to_remove_per_interval = num_to_remove_per_interval+ remainder_remove
                p_value_of_interval = prob_optimized_dict[i]['p']
                interval_to_add = (
                    interval[0], interval[0]+num_to_add_per_interval)
                new_interval_add, _, interval_remains_post = get_overap(
                    interval_to_add, interval_remain=interval)
                interval_to_remove = (interval[1]-num_to_remove_per_interval, interval[1])
                new_interval_remove, interval_remains, _ = get_overap(
                    interval_to_remove, interval_remain=interval_remains_post[0])
                
                
                new_inverse_tempered_dict[p_value_of_interval+ e_added_pos] = [new_interval_add]
                new_inverse_tempered_dict[p_value_of_interval- e_removed] = [new_interval_remove]
                new_inverse_tempered_dict[p_value_of_interval] = interval_remains
                
        print('starting the inverting process...')
        # invert the dict
        new_tempered_dict = {}
        j = 0
        for key, val in tqdm(new_inverse_tempered_dict.items()):
            sorted_val = sorted(val, key=lambda x: x[0])
            for interval in sorted_val:
                new_tempered_dict[j] = {'interval': interval, 'p': key}
                j += 1

        """
        modification in the null/zero space
        """
        assert e_added not in new_tempered_dict  # hoping
        mass_in_each_part = [(val['interval'][1] - val['interval'][0]) * val['p']
                             for key, val in new_tempered_dict.items()]
        should_be_one = np.sum(mass_in_each_part)
        new_tempered_dict[j] = {'interval': (
            U_pos, U_pos+bins_added_in_null), 'p': e_added}
        # We just add in order in the null space
        mass_in_each_part = [(val['interval'][1] - val['interval'][0]) * val['p']
                             for key, val in new_tempered_dict.items()]
        should_be_one = np.sum(mass_in_each_part)
        np.testing.assert_allclose(should_be_one, 1)
        return new_tempered_dict


# Given: U the size of the probability space
# Returns: Array with one of each element in the probability space


def genValArr(U):
    values = []
    for i in range(U):
        values.append(i)
    return values

# Given: Values which is an array with one of each element in the probability space and the updated_prob_array from the error function
# Returns: M amount of samples generated from the newly defined probability distribution


def sampleSpecificProbDist(value, probability, m):
    distrib = rv_discrete(values=(value, probability))
    new_samples = distrib.rvs(size=m)
    return new_samples


def search_for_prime(b):
    current = b
    no_prime_found = True
    while no_prime_found:
        if sympy.isprime(current):
            return current
        current += 1


def get_shuffled_index(i, base_b):
    a = 3
    base = search_for_prime(base_b)

    index_to_remove = [(i+(1))*a % base for i in range(base_b, base)]

    index = (i+(1))*a % base
    off_set = np.sum(np.array(index_to_remove) < index)
    index = index - off_set
    assert index < base_b
    return index


def scalabale_sample_distribution_with_shuffle(prob_optimized_dict, ground_truth_p, m):
    mass_in_each_part = [(val['interval'][1] - val['interval'][0]) * val['p']
                         for key, val in prob_optimized_dict.items()]
    should_be_one = np.sum(mass_in_each_part)
    size_each_regions = [(val['interval'][1] - val['interval'][0])
                         for _, val in prob_optimized_dict.items()]
    regions_to_be_merges = {}
    for key, val in prob_optimized_dict.items():
        interval_to_place = val['interval']
        for key_bigger, val in ground_truth_p.items():
            bigger_int = val['interval']
            if interval_to_place[0] >= bigger_int[0] and interval_to_place[1] <= bigger_int[1]:
                regions_to_be_merges[key] = key_bigger

    regions = list(prob_optimized_dict.keys())
    # FIRST SAMPLING, which region to sample from
    samples = sampleSpecificProbDist(regions, mass_in_each_part, m)
    index_with_regions = []
    for region in regions:
        # small error, the first number can be overwritten
        index_with_region = np.where(samples == region)[0]
        index_with_regions.append(index_with_region)
    for i, region in enumerate(regions):
        index_with_region = index_with_regions[i]
        m_in_region = index_with_region.shape[0]
        if m_in_region>0:
            interval_of_region = prob_optimized_dict[region]['interval']
            size_region = interval_of_region[1] - interval_of_region[0]
            if i in regions_to_be_merges:
                base_interval = ground_truth_p[regions_to_be_merges[i]]['interval']
                size_base_region = base_interval[1] - base_interval[0]
                offset = interval_of_region[0] - base_interval[0]
                in_region_samples = random.choices(
                    range(offset, offset+size_region), k=m_in_region)
                print('shuffling process within the samples')
                base_offset = base_interval[0]
                shuffled_index = [get_shuffled_index(
                    s, base_b=size_base_region)+base_offset for s in in_region_samples]

                samples[index_with_region] = shuffled_index
            else: # zero space, no shuffling needed
                in_region_samples = random.choices(
                    range(interval_of_region[0], interval_of_region[1]), k=m_in_region)
                samples[index_with_region] = in_region_samples
    return samples


if __name__ == '__main__':
    # This makes it so you can input U m e and b parameters when you run it in terminal.
    # This will make it easier to compare 'trials'. Will need to make a shell script

    testCase = 1
    if testCase == 1:
        if len(sys.argv) != 5:
            print("Usage:", sys.argv[0], "U m e b")
            sys.exit()

        U = int(sys.argv[1])
        m = int(sys.argv[2])
        # recall this value has been multiplied by 100 in sh script
        e = float(sys.argv[3])/100
        b = int(sys.argv[4])

        if (U or m) <= 1:
            print("U or m need to be larger than 1.")
            sys.exit()

        if (U <= m):
            print("U must be greater than m.")
            sys.exit()

        if (e or b) < 0:
            print("e or b cannot be negative.")
            sys.exit()

        if (b > 100):
            print("b must be a number in the range 0 to 100.")
            sys.exist()

    if testCase == 2:
        U = int(100)  # defining |U|, the probability space
        m = 10  # how many times uni_dist is sampled -- nice to see it go from 10k, 100k to 1M, illustrates the effectiveness working
        e = 0.1  # the total amount of error that is introduced in the probability distribution array
        b = 75  # how much of the array you would like to 'impact' or 'skew' with the error

    uni_dist = makeUniformDie(U)

    uni_prob_arr = makeUniProbArr(U)
    # plotProbDist(U, uni_prob_arr, 'elements in prob space', 'probability of occuring', 'Uniform Probability Dist') # confirmed that this works

    updated_prob_arr = errFunct(U, uni_prob_arr, e, b)
    plot_title = "Modified Probability Plotting:  U = " + \
        str(U) + " m = " + str(m) + " e = " + str(e) + " b = " + str(b)
    plotProbDist(U, updated_prob_arr, 'elements in prob space',
                 'probability of occuring', plot_title)  # adds time so removed when running lots

    val_arr = genValArr(U)

    new_samples = sampleSpecificProbDist(val_arr, updated_prob_arr, m)
    np.save("Gen_Samples", new_samples)

# TODO: Double check what convention is on the naming of python functions
