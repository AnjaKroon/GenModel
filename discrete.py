# The objective of this code is to create samples from a slightly skewed uniform probability distribution for discrete events. 

from random import uniform
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete

from time import time

# transform a prob array into a dictionnary
def prob_array_to_dict(prob_array):
    prob_hist = {}
    U = len(prob_array)
    for i in range(U):
        prob_hist[i+1] = prob_array[i]
    return prob_hist
# transform a prob dictionnary into an array
def prob_dict_to_array(prob_hist):
    prob_array = []
    all_keys = list(prob_hist.keys())
    all_keys.sort()
    for key in all_keys:
        prob_array.append(prob_hist[key])
    return prob_array

# Given: U, probability space
# Returns: np array with a uniform distribution of probability space |U|
def makeUniformDie(U): 
    sides_of_die = [None]*U #I think this part may take a lot of time, perhaps think of a faster way
    for i in range(U):
        sides_of_die[i] = i+1 
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
    x_ax = np.arange(start=1, stop=U+1, step = 1)
    plt.bar(x_ax, array, width = 1)
    #count, bins, ignored = plt.hist(array, U, density=True)
    plt.axhline(y = float(1/U), color = 'r', linestyle = '-')
    plt.xlabel(xAxis)
    plt.ylabel(yAxis)
    plt.title(title)
    plt.savefig("ModProbDist.png")


# Given: U the size of probability space
# Returns: the uniform discrete probability distribution
def makeUniProbArr(U):
    prob = 1/U
    prob_arr = []
    for i in range(U):
        prob_arr.append(prob)
    return prob_arr

# Given: U as size of probability space, array as the probability distribution (assumed uniform coming in)
# e which is the total amount of error that is introduced in the probability distribution array, and 
# percent_to_modify
# Returns: new probability distribution. 
def errFunct(U, array, e, percent_to_modify):
    # works to modify probability dist. array, works for odd U
    amt_to_modify = U*(percent_to_modify/100) # Tells us how many bins in the probability distribution we are changing

    half_point = amt_to_modify//2 # If |U| is odd, due to truncation in division, the 'extra bin' will go on the subtraction half. 
    # That means for this case, the bins_last will need to 'redistribute' how much is subtracted per bin

    bins_first = int(half_point)
    bins_last = int(amt_to_modify - half_point)
    e_per_section = e/2
    e_per_bin_first = e_per_section/bins_first #amount to add to the first section
    e_per_bin_last = e_per_section/bins_last #amount to subtract from the second section

    for i in range(bins_first):
        array[i] = array[i] + e_per_bin_first # adds same amount to first half of bins you wish to change
    
    for i in range(bins_last):
        array[bins_first+i] = array[bins_first+i] - e_per_bin_last # subtracts same amount to second half of bins you wish to change
    #print(array) # for testing
    return array 

# Given: U the size of the probability space
# Returns: Array with one of each element in the probability space
def genValArr(U):
    values = []
    for i in range(U):
        values.append(i+1)
    return values

# Given: Values which is an array with one of each element in the probability space and the updated_prob_array from the error function
# Returns: M amount of samples generated from the newly defined probability distribution
def sampleSpecificProbDist(value, probability, m):
    distrib = rv_discrete(values=(value, probability))
    new_samples = distrib.rvs(size=m)
    return new_samples

if __name__ == '__main__':
    # This makes it so you can input U m e and b parameters when you run it in terminal. 
    # This will make it easier to compare 'trials'. Will need to make a shell script
    
    testCase = 1
    if testCase == 1:
        if len(sys.argv) != 5 :
            print("Usage:", sys.argv[0], "U m e b")
            sys.exit()
        
        U = int(sys.argv[1])
        m = int(sys.argv[2])
        e = float(sys.argv[3])/100 # recall this value has been multiplied by 100 in sh script
        b = int(sys.argv[4])

        if (U or m) <= 1: 
            print("U or m need to be larger than 1.")
            sys.exit()
        
        if (U<=m):
            print("U must be greater than m.")
            sys.exit()
        
        if (e or b) < 0: # 
            print("e or b cannot be negative.")
            sys.exit()
        
        if (b>100):
            print("b must be a number in the range 0 to 100.")
            sys.exist()
        

    if testCase == 2:
        U = int(100) # defining |U|, the probability space
        m = 10 # how many times uni_dist is sampled -- nice to see it go from 10k, 100k to 1M, illustrates the effectiveness working
        e = 0.1 # the total amount of error that is introduced in the probability distribution array
        b = 75 # how much of the array you would like to 'impact' or 'skew' with the error
    
    uni_dist = makeUniformDie(U)

    uni_prob_arr = makeUniProbArr(U)
    #plotProbDist(U, uni_prob_arr, 'elements in prob space', 'probability of occuring', 'Uniform Probability Dist') # confirmed that this works

    updated_prob_arr = errFunct(U, uni_prob_arr, e, b)
    plot_title = "Modified Probability Plotting:  U = " + str(U) + " m = " + str(m) + " e = " + str(e) +  " b = " + str(b)
    plotProbDist(U, updated_prob_arr, 'elements in prob space', 'probability of occuring', plot_title) #adds time so removed when running lots

    val_arr = genValArr(U)

    new_samples = sampleSpecificProbDist(val_arr, updated_prob_arr, m)
    np.save("Gen_Samples", new_samples)

# TODO: Double check what convention is on the naming of python functions
