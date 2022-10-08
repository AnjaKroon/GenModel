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
    U_with_stair = int(U*posU/100)
    U_zero = U - U_with_stair

    # be careful to consider the case this may be fractional
    U_for_each_S = math.floor(U_with_stair/S)
    U_for_last_S = U_with_stair - (S-1)*U_for_each_S
    U_per_stairs = [U_for_each_S for i in range(S-1)]
    U_per_stairs.append(U_for_last_S)

    ratio_all_steps = list(np.arange(1, ratio, (ratio-1)/(S-1)))
    ratio_all_steps.append(ratio)
    p_first_floor = 1 / \
        (np.sum([U_per_stairs[i]*ratio_step for i,
         ratio_step in enumerate(ratio_all_steps)]))
    p_each_stair = [p_first_floor *
                    ratio_stair for ratio_stair in ratio_all_steps]
    verify_that_is_one = np.sum(
        [p_each_stair[i]*U_per_stairs[i] for i in range(S)])

    U_per_stairs.reverse()
    p_each_stair.reverse()
    current_dist = 0
    current_step = 0
    stair_histo = {}
    if U < 6**6:
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
    U = 78
    posU = 23  # pos U is % of U with > 0 pmf
    ratio = 3  # highest pmf/lowest pmf
    S = 6

    # U posU ratio and S are parameters that will define the stair function
    stair_histo = stair(U, posU, ratio, S)
    print(np.sum(list(stair_histo.values())))
    print(stair_histo)
