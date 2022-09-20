#from discrete import makeUniProbArr, errFunct, genValArr, sampleSpecificProbDist
#from gen_S import empirical_dist, genSstat
#from plot_utils import plot_S_stat
#from sampling.poisson import poisson_empirical_dist
from decimal import ROUND_DOWN
from re import L
import sys
import numpy as np
import random
import math

def p_to_bp(histo_p, U, B):
    amount_per_bin = math.floor(U/B) #3
    amount_final_bin = amount_per_bin + (U % B) #4
    new_histo = {}
    for i in range(1, B+1): # for amount of bins 1,2,3
        if i != B: # not last bin
            new_probability_for_bin = 0
            for j in range(1, amount_per_bin+1): 
                new_probability_for_bin = new_probability_for_bin + histo_p.get(j) # returns the probability at j
            new_histo[i] =new_probability_for_bin
        if i == B: # last bin
            final_bin_probability = 0
            for j in range(1, amount_final_bin+1):
                final_bin_probability = final_bin_probability + histo_p.get(j)
            new_histo[i] = final_bin_probability
    return new_histo

def transform_samples(b_p, histo_p, p_samples, U, B):
    # define subdivisions
    print("size of p_samples is ", len(p_samples))
    amount_per_bin = math.floor(len(p_samples)/B) 
    amount_final_bin = amount_per_bin + (len(p_samples) % B)
    # bins = B
    new_samples = []
    #for item in histo_p.items():
        # if sample is from 0 to 33, add element [1] to new_samples
        # if sample is from 34 to 66, add element [2] to new_samples
        # if sample is from 67 to 100, add element [3] to new_samples
    for i in range(1, B+1):
        if i != B:
            for amt in range(amount_per_bin):
                new_samples.append(i)
        if i == B:
            for amt in range(amount_final_bin):
                new_samples.append(i)
    
    print(new_samples)
            
    # for i amount of bins
        # # i = 1 lets say
        # if not last bin, let's add numbers, for the amount_per_bin
        # if last bin, let's add numbers, for the amount_final_bin
    return new_samples

if __name__ == '__main__':
    sample_histo = {1:0.1, 2:0.1, 3:0.1, 4:0.1, 5:0.1, 6:0.1, 7:0.1, 8:0.1, 9:0.1, 10:0.1}
    sample_b = {1: 0.33, 2: 0.33, 3:0.33}
    p_samples = [1,2,3,3,4,5,6,7,8,9,10,1,2,4,5,6,7,8,9,10]
    
    
    U = 10
    B = 3



    b_p = p_to_bp(sample_histo, U, B) # works
    b_out = transform_samples(b_p, sample_histo, p_samples, U, B) # works

    # the question is how to get the relevant things in
    # need some sort of histo for the transform_samples -- there should be an array of the poissonized samples
