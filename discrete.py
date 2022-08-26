from random import uniform
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint, rv_discrete
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html

# Creates a uniform distribution of probability space |U|
# Creates one number of each value starting at 1 ... U
# Returns a np array
def uniform_die(U):
    sides_of_die = [None]*U # I think this part may take a lot of time, perhaps think of a faster way
    for i in range(U):
        sides_of_die[i] = i+1 
    return sides_of_die

# ArrayToSample is the array you would like to sample
# m is the amount of times you would like to sample
# Returns a numpy array with the samples in it
def sample(ArrayToSample, m):
    samples = np.random.choice(ArrayToSample, size=m)
    return samples

# array is a np array
# does not return anything but displays a graph
def plot(array, bins, xAxis, yAxis, title):
    count, bins, ignored = plt.hist(array, bins, density=True)
    #plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.xlabel(xAxis)
    plt.ylabel(yAxis)
    plt.title(title)
    plt.show()

#TODO: Still need to get an array with the probability distributions. The current array being passed in is solely 
# We can make one ourselves by stating that each item has probability distribution of 1/|U| 
# # OR find a function that takes in an array U and gives us the probability distribution? But it should always be uniform at that point...
# Okay so let's assume we make an array [1,2,3, ... 10] with probability for each element being [0.1, 0.1, 0.1 ... 0.1] -- i.e. uniform
# Now we go through this err_function and get a new probability for each element that we want to impose on the origional distribution -- [0.101, 0.101, ... 0.999, 0.999]
# I think this is possible because I recall a way to use a library and force a probabilty distribution on an array 
def err_funct(U, array, e, percent_to_modify):
    # U is an integer that is the size of the probability space
    # array would be something like [0.1 0.1 0.1 ...] representing the uniform probability distribution if U was 10
    # e would be how much we want to skew the probability distribution -- something like 10 %
    # percent_to_modify would be how much of the probability distribution to modify -- 100 is all of the prob dist, 50 would be half of it etc.

    # The intuition:
    # if percent_to_modify is 100%, you take the WHOLE array, split it into two, increase the first half by e/|U| and decrease the second half by e/|U|
    # if percent_to_modify is 50%, you take HALF of the array (U/2), split that into two parts, increase the first half by e/HALF, decrease the second half by e/HALF
    # if percent_to_modify is 20%, multiply array size by .2 and that is now the amount of die sides you will modify the weight of

    amt_to_modify = U*(percent_to_modify/100) # Tells us how many bins in the probability distribution we are changing

    half_point = amt_to_modify//2 # due to truncation, the 'extra bin' if |U| is odd will go on the subtraction half

    bins_first = half_point
    bins_last = U - half_point

    for i in range(bins_first):
        # you want to modify based on the amount of bins so its not dependent on being even
        # like this one should be fine
        array[i] = array[i] + e/amt_to_modify # adds same amount to first half
    
    for i in range(bins_last):
        # but the amount I am subtracting here is not if |U| is odd
        array[bins_last+i] = array[bins_last+i] - e/amt_to_modify
    
    # CURRENTLY STILL SETUP SUCH THAT |U| NEEDS TO BE EVEN


if __name__ == '__main__':
    U = 100 # defining |U|, the probability space -- RIGHT NOW |U| NEEDS TO BE EVEN
    m = 20 # how many times uni_dist is sampled
    e = 0.1

    # create "die" with |U| sides
    uni_dist = uniform_die(U)

    # sample the array m times
    orig_samples = sample(uni_dist, m)

    # let's plot the uniform probability distribution. 
    plot(uni_dist, U, 'probability space', 'probability of event occuring', 'Uniform Probability Example')

    # now let's introduce the "error". 
    # half_half_weighted_uni_dist = error_function(U, uni_dist, 0.1, 100)


