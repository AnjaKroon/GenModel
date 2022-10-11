# The objective of this code is to create samples from a slightly skewed uniform probability distribution for discrete events.

from random import uniform
import random
import sys
from tkinter import EXCEPTION
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete

from time import time

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

# Given: U as size of probability space, array as the probability distribution (assumed uniform coming in)
# e which is the total amount of error that is introduced in the probability distribution array, and
# percent_to_modify
# Returns: new probability distribution.
# percent_to_modify_null is the percentage of error to be situaded in zero space


def errFunct(U, init_array, e, percent_to_modify, percent_to_modify_null=0):

    array = np.copy(init_array)  # copy to avoid modifying the passed array

    U_pos = np.where(array)[0].shape[0]  # count the positive space

    # first, we compute the error on the positive space
    percent_pos_space = percent_to_modify-percent_to_modify_null
    assert percent_pos_space >= 0
    # works to modify probability dist. array, works for odd U
    # Tells us how many bins in the probability distribution we are changing
    amt_to_modify_pos = U_pos*(percent_pos_space/100)
    # on the null space, we can only add error.
    amt_to_modify_null = (U-U_pos) * (percent_to_modify_null/100)

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
            raise EXCEPTION
        # subtracts same amount to second half of bins you wish to change
        array[i] = array[i] - e_removed

    # this check that the array sum up to one, to be a valid prob. I use assert close because something it won't be excatly one because of numerical error.
    should_be_one = np.sum(array)
    np.testing.assert_allclose(should_be_one, 1)
    return array

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


NOT_TO_BIG = 10000


# def find_bigger_divisor(U):
#     # first we find how deep the tree needs to be. (How many sampling step will be needed)
#     tree_is_too_wide = True
#     tree_depht = 2
#     while tree_is_too_wide:
#         tree_wideness = U**(1/tree_depht)
#         if tree_wideness < NOT_TO_BIG:
#             tree_is_too_wide = False
#         tree_depht += 1
#     print('the sampling tree is', tree_depht)
#     # Then, we find even divisor

# NOTDONE TODO scalable
def scalabale_sample_distribution(U, function_prob, m, flatten_dist=None):
    size_subsampling_space = NOT_TO_BIG
    probability = [
        1/size_subsampling_space for _ in range(size_subsampling_space)]
    values = list(range(size_subsampling_space))
    if flatten_dist is None:  # we can assume that the distribution is uniform, so we can split the space however we like
        first_split_space = int(U / size_subsampling_space)
        assert first_split_space * size_subsampling_space == U  # we need this for now
        second_split_space = int(first_split_space / size_subsampling_space)
        assert second_split_space * \
            size_subsampling_space == first_split_space  # we need this for now

        distrib = rv_discrete(values=(values, probability))
        sample_first_space = distrib.rvs(size=m)
        sample_second_space = distrib.rvs(size=m)

        probability = [1/second_split_space for _ in range(second_split_space)]
        values = list(range(second_split_space))
        distrib_reminder = rv_discrete(values=(values, probability))
        sample_reminder = distrib_reminder.rvs(size=m)

        # build the samples from the tree sampling scheme
        samples = [m*size_subsampling_space *
                   second_split_space for m in sample_first_space]
        samples = [m+(sample_second_space[i]*second_split_space)
                   for i, m in enumerate(samples)]
        samples = [m+(sample_reminder[i]) for i, m in enumerate(samples)]
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
