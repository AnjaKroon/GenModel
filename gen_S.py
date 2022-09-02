# objective of this code is to generate the S statistic according to algorithm 1
from math import log, sqrt
import numpy as np
import sys
from numpy import savetxt

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
    
    histo_with_zeros = {}
    for i in range(incoming_U):
        i = i+1 # offset by 1 because histo starts at 1 not 0
        if not isinstance(histo.get(i), int): # basically, if it is not in the samples, you want to assign it to a value of zero with a key corresponding to i
            histo_with_zeros.update({i: 0})
        else: # else you would add the histo type and value to the dictionary
            histo_with_zeros.update({i: histo.get(i)})
    return histo_with_zeros
        
if __name__ == '__main__':
    # FOR TESTING EITHER COMMENT THIS SECTION OUT OR THE NEXT TESTING SECTION
    # U, m, e, and b 
    #incoming_arr_samples = np.load('Gen_Samples.npy')
    #if len(sys.argv) != 5 :
    #    print("Usage:", sys.argv[0], "U m e b")
    #    sys.exit()
    #path1 = sys.argv[1]
    #path2 = sys.argv[2]
    #path3 = sys.argv[3]
    #path4 = sys.argv[4]  
    #U = int(path1)
    #m = int(path2)
    #e = float(path3)/100 # recall this value has been multiplied by 100 in sh script
    #b = int(path4)

    # FOR TESTING EITHER COMMENT THIS SECTION OUT OR THE PREV TESTING SECTION
    incoming_arr_samples = [1,2,3,3,3,4,4,5,6,7,8,9,10,11,1,14,17]
    U = 20
    m = 17
    e = 0.1
    b = 100

    p_emp_dependent = empirical_dist(U, m, incoming_arr_samples)

    new_lis = list(p_emp_dependent.items())
    two_dim_arr = np.array(new_lis)
    print(two_dim_arr)
    e=e*100
    savetxt(f'histo_{U}_{m}_{e}_{b}.csv', two_dim_arr, delimiter=',', fmt='%d', header='item in prob space, frequency in samples generated')

    #TODO: would also maybe be useful to plot your histogram to check
    #TODO: also need to remove dependencies to make the histogram independent? "sample the samples"? need to follow up on the procedure for that


