from discrete import makeUniProbArr, errFunct, genValArr, sampleSpecificProbDist
from gen_S import empirical_dist, genSstat
import numpy as np
import random

def sampling_down_m(max_m, sampled_m, count_of_max_m):
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
        count_of_sampled_m = sampling_down_m(max_m, sampled_m, count_of_max_m)

        pois_empirical_pmf[positive_support] = count_of_sampled_m/sampled_m
    return pois_empirical_pmf


if __name__ == '__main__':

    U = 1000
    m = 10
    e = 0.1  # recall this value has been multiplied by 100 in sh script
    b = 100

    uni_prob_arr = makeUniProbArr(U)
    new_samples = sampleSpecificProbDist(genValArr(U), uni_prob_arr, m)
    p_emp_dependent = empirical_dist(U, m, new_samples)
    s_statistic = genSstat(p_emp_dependent, U)
    print("S of uniform", s_statistic)

    
    p_emp_dependent = poisson_empirical_dist(
    U, m, new_samples, lambda m: sampleSpecificProbDist(genValArr(U), uni_prob_arr, m))
    s_statistic = genSstat(p_emp_dependent, U)
    print("S of uniform with poisson", s_statistic)

    updated_prob_arr = errFunct(U, uni_prob_arr, e, b)
    new_samples = sampleSpecificProbDist(genValArr(U), updated_prob_arr, m)
    p_emp_dependent = empirical_dist(U, m, new_samples)
    s_statistic = genSstat(p_emp_dependent, U)
    print("S of tempered", s_statistic)

    # U m e b "S of uniform": NUMBER "S of uniform with poisson":NUMBER "S of tempered":NUMBER

    
