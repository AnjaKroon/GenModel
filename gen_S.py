# objective of this code is to generate the S statistic according to algorithm 1
from math import log, sqrt
import numpy as np

def empirical_dist(incoming_U, incoming_m, incoming_arr_samples):
    # Create a dictionary with size U. Note index will be shifted by 1 as i starts at 0 and numbers start at 1
    key = []
    for i in range(incoming_m):
        key = np.append(key, incoming_arr_samples[i]) 
    value = 0
    histo = dict.fromkeys(key, int(value))  # Note that the keys are now floats and you may want to change them to integers later but just beware
    for i in range(len(incoming_arr_samples)):  # for each value in the samples array, add +1 to the frequency that the value corresponds to in histo
        val = histo.get(incoming_arr_samples[i])
        histo.update({incoming_arr_samples[i]:(val+1)})
    return histo
        
if __name__ == '__main__':
    incoming_arr_samples = np.load('Gen_Samples_100_10_100.npy')
    incoming_U = 10
    incoming_m = len(incoming_arr_samples)

    p_emp_dependent = empirical_dist(incoming_U, incoming_m, incoming_arr_samples)

    print (p_emp_dependent)


