from random import uniform
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint, rv_discrete

# Given: U, probability space
# Returns: a np array with one number of each value starting at 1 ... U -- i.e. a uniform distribution of probability space |U|
def makeUniformDie(U):
    sides_of_die = [None]*U # I think this part may take a lot of time, perhaps think of a faster way
    for i in range(U):
        sides_of_die[i] = i+1 
    return sides_of_die

# Given: ArrayToSample is the array you would like to sample, m is the amount of times you would like to sample
# Returns: a numpy array with the samples in it
def sampleAnArray(ArrayToSample, m):
    samples = np.random.choice(ArrayToSample, size=m)
    return samples

# Given: array is a np array to plot, bins is number of categories to plot, xAxis label, yAxis label, and title label
# Returns: a bar graph displayed
def plot(array, bins, xAxis, yAxis, title):
    count, bins, ignored = plt.hist(array, bins, density=True)
    #plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.xlabel(xAxis)
    plt.ylabel(yAxis)
    plt.title(title)
    plt.show()

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

    half_point = amt_to_modify//2 # If |U| is odd, due to truncation in division, the 'extra bin' will go on the subtraction half. That means for this case, the bins_last will need to 'redistribute' how much is subtracted per bin

    bins_first = int(half_point)
    bins_last = int(U - half_point)
    e_per_section = e/2
    e_per_bin_first = e_per_section/bins_first #amount to add to the first section
    e_per_bin_last = e_per_section/bins_last #amount to subtract from the second section

    for i in range(bins_first):
        array[i] = array[i] + e_per_bin_first # adds same amount to first half of bins you wish to change
    
    for i in range(bins_last):
        array[bins_first+i] = array[bins_first+i] - e_per_bin_last # subtracts same amount to second half of bins you wish to change
    
    print(array) # for testing
    return array 

def genValArr(U):
    # Given: U the size of the probability space
    # Returns: Array with one of each element in the probability space
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
    U = int(9) # defining |U|, the probability space
    m = 100000 # how many times uni_dist is sampled -- nice to see it go from 10k, 100k to 1M, illustrates the effectiveness working
    e = 0.1 # the total amount of error that is introduced in the probability distribution array
    b = 100 # how much of the array you would like to 'impact' or 'skew' with the error

    uni_dist = makeUniformDie(U) # creates a uniform 'die' with |U| sides
    plot(uni_dist, U, 'probability space', 'probability of event occuring', 'Uniform Probability Example') # visual confirmation of uniform dist

    uni_prob_arr = makeUniProbArr(U)

    # orig_samples = sampleAnArray(uni_dist, m) # optional, used for testing

    updated_prob_arr = errFunct(U, uni_prob_arr, e, b)

    val_arr = genValArr(U)

    new_samples = sampleSpecificProbDist(val_arr, updated_prob_arr, m)

    plot(new_samples, U, 'probability space', 'probability of event occuring', 'Modified Probability Plotting')
