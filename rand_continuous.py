from random import uniform
import numpy as np
import matplotlib.pyplot as plt

# Uniform Sample Generation
# Low (float) is the low value of samples generated. High (float) is the high value of samples generated. Size (integer) is the amount of samples generated.
# Returns a numpy array with the uniformly distributed samples.
def uniform_sample_gen(low, high, size):
    uni_samples = rng.uniform(low, high, size)
    return uni_samples

# Normal Sample Generation
# Low (float) and high (float) are ranges for the samples generated. Size (integer) is the amount of samples generated.
# Returns a numpy array with the normaly distributed samples. 
def normal_sample_gen(low, high, size):
    norm_samples = rng.normal(low, high, size)
    for i in range(size):
        norm_samples[i] = norm_samples[i]*(0.333333333333333)
    return norm_samples

# array is a np array. bins are the amount of categories the plot will show
# does not return anything but displays a graph
def plot(array, bins):
    count, bins, ignored = plt.hist(array, bins, density=True)
    plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.show()

if __name__ == '__main__':
    # Sets a seed for the random number generator
    rng = np.random.default_rng(12345)

    min_value = 0.0
    max_value = 1.0
    samples = 1000

    uniform = uniform_sample_gen(min_value, max_value, samples)
    normal = normal_sample_gen(min_value, max_value, samples)

    # Checking that the output is within the desired range for normal
    if ((np.all(normal >= min_value)) and (np.all(normal < max_value))):
        print("Samples within range for normal distribution") #this is not printing
        # somehow, I still have a few points that are above 1.0 even with compensation that should have fixed it?
        # It's definitely something in stats I am forgetting about
    
    # Checking that the output is within the desired range for uniform
    if ((np.all(uniform >= min_value)) and (np.all(uniform < max_value))):
        print("Samples within range for uniform distribution")

    #print(uniform)
    plot(uniform, 16)
    #print(normal)
    plot(normal, 16)

    
    