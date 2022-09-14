import matplotlib.pyplot as plt
from matplotlib import markers
import numpy as np
import matplotlib
import os
import matplotlib.colors as mcolors
import math
import random
from scipy.stats import bootstrap
matplotlib.rcParams['text.usetex'] = True
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 17}

matplotlib.rc('font', **font)
cycle = list(mcolors.TABLEAU_COLORS.values())
random.shuffle(cycle)


def get_color(i):
    if i >= len(cycle):
        return cycle[i % len(cycle)]
    else:
        return cycle[i]

def get_ci(trials):
    ci = bootstrap((trials,), np.mean, confidence_level=0.9,random_state=0)
    return ci.confidence_interval


def plot_S_stat(x, dict_y, title):

    for i, (key, all_trials) in enumerate(dict_y.items()):
        color = get_color(i)
        y = [np.mean(trials) for trials in all_trials]
        all_ci = [get_ci(trials) for trials in all_trials] 
        plt.plot(x, y, color=color, label=key)
        ci_over = [ci.high for i, ci in enumerate(all_ci)]
        ci_under = [ci.low for i, ci in enumerate(all_ci)]
        plt.fill_between(x, ci_under, ci_over, color=color, alpha=.1)
    plt.legend()
    plt.xlabel('$|U|$')
    plt.ylabel('$S$')
    plt.tight_layout()
    plt.savefig(title)
