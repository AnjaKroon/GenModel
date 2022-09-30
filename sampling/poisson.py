import random 
import numpy as np

"""
Poissonization code. The poisson_empirical_dist 'poissonize' an array of samples.
"""

# Helper function for poisson_empirical_dist
def _sampling_down_m(max_m, sampled_m, count_of_max_m):
    number_of_elements_to_be_dropped = max_m-sampled_m
    if number_of_elements_to_be_dropped == 0:
        return count_of_max_m
    index_to_drop = random.sample(
        range(max_m), number_of_elements_to_be_dropped)
    num_to_drop = 0
    for c in range(count_of_max_m):
        if c in index_to_drop:
            num_to_drop += 1
    return count_of_max_m-num_to_drop

# Takes size of space U, size of samples m, samples, and the function that can generate more samples if needed.

def poisson_empirical_dist(U, m, incoming_arr_samples, sample_func_for_additional):
    # sample a list of U number of samples from Poisson
    all_random_m = np.random.poisson(m, U)
    # get the max number of samples to compute the number of missing samples
    max_m = int(max(all_random_m))

    # sample the missing additional samples
    additional_samples = sample_func_for_additional(max_m-m)
    arr_samples = np.concatenate((incoming_arr_samples, additional_samples))

    # getting the histogram of the list of samples
    histogram_samples = {}
    for sample in arr_samples:
        sample_key = int(sample)
        if sample_key in histogram_samples:
            histogram_samples[sample_key] += 1
        else:
            histogram_samples[sample_key] = 0

    # building the empitical pmf
    pois_empirical_pmf = {}
    for i, positive_support in enumerate(histogram_samples.keys()):
        sampled_m = int(all_random_m[i])  # get the random number of samples
        count_of_max_m = histogram_samples[positive_support]
        count_of_sampled_m = _sampling_down_m(max_m, sampled_m, count_of_max_m)

        pois_empirical_pmf[positive_support] = count_of_sampled_m/sampled_m
    # normalize everything back to get a valid pmf
    N = np.sum(list(pois_empirical_pmf.values()))
    for key, val in pois_empirical_pmf.items():
        pois_empirical_pmf[key] = val/N
    N = np.sum(list(pois_empirical_pmf.values())) # this should be one
    return pois_empirical_pmf
