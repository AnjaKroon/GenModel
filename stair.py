from random import uniform
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete
import math
from time import time

def stair(U, posU, ratio, S):
    # From my understanding
    # Take U * posU to get the amount of U that will have a stair function
    # S is the total amount of steps thus (U*posU)/S is the amount of U that each step will take
    # ratio is highest pmf/lowest pmf -- representative of the amount of "y step" in between each stair
    # pmf of each value for the whole U will have to sum to 1
    # highest pmf / lowest pmf = ratio -- (?/common denom) / (?/common denom)
    U_with_stair = int(posU * U)
 

    middle_step = math.floor(S/2)
    dist_middle_step = 1/U_with_stair
    stair_histo = {}
    if U <= 7**7:
        for i, size_stair in enumerate(U_per_stairs):
            current_dist = p_each_stair[i]
            for _ in range(size_stair):
                current_dist = p_each_stair[i]
                stair_histo[current_step] = current_dist
                current_step += 1
       
    else:
        start_interval = 0
        for i, size_stair in enumerate(U_per_stairs):
            current_dist = p_each_stair[i]
            interval = [start_interval, start_interval+size_stair]
            stair_histo[i] = {'interval': interval, 'p': current_dist}
            start_interval += size_stair
    return stair_histo


if __name__ == '__main__':
    U = 6
    posU = 1 # pos U is % of U with > 0 pmf
    ratio = 3 # highest pmf/lowest pmf
    S = 3
    # U posU ratio and S are parameters that will define the stair function
    stair_histo = stair(U, posU, ratio, S)
    print (stair_histo)