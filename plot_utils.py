import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
from scipy.stats import bootstrap
# sketchy way of figuring out if latex is installed, might fail.
# If it does, comment out
from distutils.spawn import find_executable
if find_executable('latex'):
    matplotlib.rcParams['text.usetex'] = True

font = {'family': 'normal',
        'weight': 'bold',
        'size': 17}

matplotlib.rc('font', **font)
cycle = list(mcolors.TABLEAU_COLORS.values())


def get_color(i):
    if i >= len(cycle):
        return cycle[i % len(cycle)]
    else:
        return cycle[i]


def get_ci(trials):
    ci = bootstrap((trials,), np.mean, confidence_level=0.9, random_state=0)
    return ci.confidence_interval


def plot_S_stat(x, dict_y, title, xlabel, ylabel):

    for i, (key, val) in enumerate(dict_y.items()):
        color = get_color(i)
        if type(val[0]) is list:
            all_trials = val
            y = [np.mean(trials) for trials in all_trials]
            all_ci = [get_ci(trials) for trials in all_trials]
            plt.plot(x, y, color=color, label=key)
            ci_over = [ci.high for i, ci in enumerate(all_ci)]
            ci_under = [ci.low for i, ci in enumerate(all_ci)]
            plt.fill_between(x, ci_under, ci_over, color=color, alpha=.1)
        else:
            plt.plot(x, val, color=color, label=key)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(title)
    plt.close()
