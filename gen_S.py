# objective of this code is to generate the S statistic according to algorithm 1
from math import log, sqrt
import numpy as np
import sys
from numpy import savetxt
import pandas as pd
import scipy


def empirical_dist(incoming_U, incoming_m, incoming_arr_samples):
    # Create a dictionary with size U. Note index will be shifted by 1 as i starts at 0 and numbers start at 1
    key = []
    for i in range(incoming_m):
        key = np.append(key, incoming_arr_samples[i])
    value = 0
    histo = dict.fromkeys(key, int(value))
    # for each value in the samples array, add +1 to the frequency that the value corresponds to in histo
    for i in range(len(incoming_arr_samples)):
        val = histo.get(incoming_arr_samples[i])
        histo.update({incoming_arr_samples[i]: (val+1)})
    # Adding the zeros into the histogram
    # FLO: this is not necessary, the best is to have the non existing entries as begin 0 by default (to save memory)
    histo_with_zeros = {}
    for i in range(incoming_U):
        i = i+1  # offset by 1 because histo starts at 1 not 0
        # basically, if it is not in the samples, you want to assign it to a value of zero with a key corresponding to i
        if not isinstance(histo.get(i), int):
            histo_with_zeros.update({i: 0})
        else:  # else you would add the histo type and value to the dictionary
            histo_with_zeros.update({i: histo.get(i)/incoming_m})
    p_emp = histo_with_zeros
    return p_emp


def intoCSV(arr, U, m, e, b):
    two_dim_arr = np.array(list(arr.items()))
    DF = pd.DataFrame(two_dim_arr)
    e = int(e*100)
    DF.to_csv(f'histo_{U}_{m}_{e}_{b}.csv', index=False, header=False)
    return


def test_to_reject_chi_square(uni_prob_array, p_emp_array):
    a = np.sum(uni_prob_array)
    b =  np.sum(p_emp_array)
    chi_square_out = scipy.stats.chisquare(uni_prob_array, p_emp_array)
    p_value = chi_square_out[1]
    if p_value < 0.95:
        reject = True
    else:
        reject = False
    return reject


def genSstat(dictionary, U):
    sum = 0
    inv_U = 1/U
    for i in range(U):
        if i+1 in dictionary:
            sum = sum+(abs(dictionary.get(i+1)-inv_U))
        else:
            sum = sum+inv_U
    sum = sum/2
    return sum


if __name__ == '__main__':
    testCase = 1  # should be 1 or 2 depending on whether you want to run the program standalone or with a .sh script
    if testCase == 1:
        incoming_arr_samples = np.load('Gen_Samples.npy')
        if len(sys.argv) != 5:
            print("Usage:", sys.argv[0], "U m e b")
            sys.exit()
        U = int(sys.argv[1])
        m = int(sys.argv[2])
        # recall this value has been multiplied by 100 in sh script
        e = float(sys.argv[3])/100
        b = int(sys.argv[4])

    if testCase == 2:
        incoming_arr_samples = [2, 3, 3, 3, 4,
                                4, 5, 6, 7, 8, 9, 10, 11, 1, 14, 17]
        U = 19
        m = 16
        e = 0.1
        b = 100

    p_emp_dependent = empirical_dist(U, m, incoming_arr_samples)
    intoCSV(p_emp_dependent, U, m, e, b)  # Turning into .csv file
    s_statistic = genSstat(p_emp_dependent, U)
    print(U, " ",  m, " ", e, " ", b, " ", s_statistic)

    # TODO: also need to remove dependencies to make the histogram independent? "sample the samples"? need to follow up on the procedure for that
