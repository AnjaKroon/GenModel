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
    U_with_stair = U*posU
    U_for_each_S = U_with_stair/S # be careful to consider the case this may be fractional

    middle_step = math.floor(S/2)
    dist_middle_step = 1/U_with_stair
    stair_histo = {}

    # DIST_DELTA is the standardized value in which the pmf increments between each step
    highest = ratio
    lowest = 1
    DIST_DELTA = ((highest-lowest)/(S-1))/U_with_stair

    current_step = middle_step
    current_dist = dist_middle_step
    while (current_step>-1):
        stair_histo[current_step] =  current_dist
        current_dist = current_dist + DIST_DELTA
        current_step = current_step - 1 # to iterate though the while loop/the left half of the steps
    
    current_step = middle_step # reset the value of current step
    current_dist = dist_middle_step # reset the value of the current distribution
    current_step = current_step +1 # I don't want to overwrite the middle step. I want to start at the next one
    while (current_step<S):
        current_dist = current_dist - DIST_DELTA
        stair_histo[current_step] = current_dist
        current_step = current_step +1 # to iterature through the while loop/the right half of the steps
    
    # to check


    return stair_histo


if __name__ == '__main__':
    U = 6
    posU = 1 # pos U is % of U with > 0 pmf
    ratio = 3 # highest pmf/lowest pmf
    S = 3
    # U posU ratio and S are parameters that will define the stair function
    stair_histo = stair(U, posU, ratio, S)
    print (stair_histo)