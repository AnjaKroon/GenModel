from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
from scipy.stats import bootstrap
import random as rd
# sketchy way of figuring out if latex is installed, might fail.
# If it does, comment out
from distutils.spawn import find_executable
from itertools import cycle
if find_executable('latex'):
    matplotlib.rcParams['text.usetex'] = True

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}


matplotlib.rc('font', **font)
cycle = list(mcolors.TABLEAU_COLORS.values())
# cycle = list(mcolors.CSS4_COLORS.values())


def get_color(i):
    if i >= len(cycle):
        return cycle[i % len(cycle)]
    else:
        return cycle[i]


def get_ci(trials):
    ci = bootstrap((trials,), np.mean, confidence_level=0.9, random_state=0)
    return ci.confidence_interval


def plot_stat(title, xlabel, ylabel):
    # my hunch is to wrap this in another for loop -- need to make sure it still runs without it
    # if there is only one m you want to work with, you need to provide a dictionary with that singe m in it -- constraint
    # print("x", x)
    # print("dict_y", dict_y)
    # put_on_plot(x, dict_y)

    SMALL_SIZE = 5
    matplotlib.rcParams.update({'font.size': 14})
    
    
    '''
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=5)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
    '''

    # idea to break this up here....
    # essentially invoke part 1 multiple times and then invoke part 2 as below
    plt.legend()
    plt.xticks([4,5,6,7,8], ["4","5","6","7","8"])
    plt.xlabel(xlabel)
    
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(title)
    plt.close()


# x = [1,2,3] dict_y = {label:[[3,3,2],[4,5,4],[5,5,5]]}
def put_on_plot(x, dict_y, label_dict=None):
    for i, (key, val) in enumerate(dict_y.items()):
        if label_dict is not None:
            key = label_dict[key]
        color = get_color(i)
        if type(val[0]) is list and len(val[0]) > 1:
            all_trials = val
            y = [np.mean(trials) for trials in all_trials]

            plt.plot(x, y, color=color, label=key)
            try:
                all_ci = [get_ci(trials) for trials in all_trials]
                ci_over = [ci.high for i, ci in enumerate(all_ci)]
                ci_under = [ci.low for i, ci in enumerate(all_ci)]
                plt.fill_between(x, ci_under, ci_over, color=color, alpha=.1)
            except:
                print('couldnt do the ci', all_trials[0])
        else:
            print(x)
            print(val)
            plt.plot(x, val, color=color, label=key)

    # idea to break this up here....
    # essentially invoke part 1 multiple times and then invoke part 2 as below
