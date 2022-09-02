# objective of this code is to generate the S statistic according to algorithm 1
from math import log, sqrt
import numpy as np
import sys
from numpy import savetxt
import pandas as pd

def empirical_dist(incoming_U, incoming_m, incoming_arr_samples):
    # Create a dictionary with size U. Note index will be shifted by 1 as i starts at 0 and numbers start at 1
    key = []
    for i in range(incoming_m):
        key = np.append(key, incoming_arr_samples[i]) 
    value = 0
    histo = dict.fromkeys(key, int(value)) 
    for i in range(len(incoming_arr_samples)):  # for each value in the samples array, add +1 to the frequency that the value corresponds to in histo
        val = histo.get(incoming_arr_samples[i])
        histo.update({incoming_arr_samples[i]:(val+1)})
    # Adding the zeros into the histogram
    histo_with_zeros = {}
    for i in range(incoming_U):
        i = i+1 # offset by 1 because histo starts at 1 not 0
        if not isinstance(histo.get(i), int): # basically, if it is not in the samples, you want to assign it to a value of zero with a key corresponding to i
            histo_with_zeros.update({i: 0})
        else: # else you would add the histo type and value to the dictionary
            histo_with_zeros.update({i: histo.get(i)/incoming_m})
    p_emp = histo_with_zeros
    return p_emp

def intoCSV(arr, U, m, e, b):
    two_dim_arr = np.array(list(arr.items()))
    DF = pd.DataFrame(two_dim_arr) 
    e= int(e*100)
    DF.to_csv(f'histo_{U}_{m}_{e}_{b}.csv')
    df = pd.read_csv(f'histo_{U}_{m}_{e}_{b}.csv')
    df = df.drop([df.columns[0]], axis=1)
    df = df.iloc[1: , :]
    df.to_csv(f'histo_{U}_{m}_{e}_{b}.csv', index=False)
    return

def genSstat(dictionary, U):
    sum = 0
    inv_U = 1/U
    for i in range(U):
        # print(dictionary.get(i+1))
        sum = sum+(abs(dictionary.get(i+1)-inv_U))
    sum = sum/2
    return sum
        
if __name__ == '__main__':
    # FOR TESTING EITHER COMMENT THIS SECTION OUT OR THE NEXT TESTING SECTION
    # U, m, e, and b 
    incoming_arr_samples = np.load('Gen_Samples.npy')
    if len(sys.argv) != 5 :
        print("Usage:", sys.argv[0], "U m e b")
        sys.exit()
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    path3 = sys.argv[3]
    path4 = sys.argv[4]  
    U = int(path1)
    m = int(path2)
    e = float(path3)/100 # recall this value has been multiplied by 100 in sh script
    b = int(path4)

    # FOR TESTING EITHER COMMENT THIS SECTION OUT OR THE PREV TESTING SECTION
    #incoming_arr_samples = [1,2,3,3,3,4,4,5,6,7,8,9,10,11,1,14,17]
    #U = 20
    #m = 17
    #e = 0.1
    #b = 100

    p_emp_dependent = empirical_dist(U, m, incoming_arr_samples)
    intoCSV(p_emp_dependent, U, m, e, b) # Turning into .csv file
    s_statistic = genSstat(p_emp_dependent, U)
    print(s_statistic)
    
    #TODO: would also maybe be useful to plot your histogram to check
    #TODO: also need to remove dependencies to make the histogram independent? "sample the samples"? need to follow up on the procedure for that


