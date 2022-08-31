# objective of this code is to generate the S statistic according to algorithm 1
from math import log, sqrt
import numpy as np

def computeMinError(U, delta, m):
    e_star = sqrt(sqrt(U*log(1/delta) + log(1/delta))/m)
    return e_star

if __name__ == '__main__':
    C = 1.0
    delta = 0.05

    # to be pulled from user later
    U = 1000
    m = 100
    # p_hat = discrete.py(new_samples) or something of the sort

    e_star = computeMinError(U, delta, m) # with calculator and givens, should return 0.6 for e*
    print(e_star) # returns 0.74 and needs to return 0.6 approx.

    # for step 3 in algo 1, don't we have that already from discrete.py?


