
import matplotlib.font_manager as font_manager
import numpy as np
import matplotlib.pyplot as plt
from distutils.spawn import find_executable
import matplotlib
if find_executable('latex'):
    matplotlib.rcParams['text.usetex'] = True


# set width of bar
barWidth = 0.07
fig = plt.subplots(figsize=(12, 8))
font = {'family': 'normal',
        'weight': 'normal',
        'size': 22}


matplotlib.rc('font', **font)
# set height of bar
t1 = [3, 4, 5, 6]
t2 = [3, 4, 5, 6]
t3 = [3, 4, 5, 6]
t4 = [3, 4, 5, 6]
t5 = [3, 4, 5, 6]
t6 = [3, 4, 5, 6]
t7 = [3, 4, 5, 6]
t8 = [3, 5, 5, 6]
t9 = [3, 5, 5, 6]
t10 = [3, 5, 5, 6]
# Set position of bar on X axis
br1 = np.arange(len(t1))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
br6 = [x + barWidth for x in br5]
br7 = [x + barWidth for x in br6]
br8 = [x + barWidth for x in br7]
br9 = [x + barWidth for x in br8]
br10 = [x + barWidth for x in br9]
plt.axhline(y = 3, color = 'k', linestyle = '-')
# Make the plot
plt.bar(br1, t1, color='r', width=barWidth,
        edgecolor='grey')
plt.bar(br2, t2, color='r', width=barWidth,
        edgecolor='grey')
plt.bar(br3, t3, color='r', width=barWidth,
        edgecolor='grey')
plt.bar(br4, t4, color='r', width=barWidth,
        edgecolor='grey')
plt.bar(br5, t5, color='r', width=barWidth,
        edgecolor='grey')
plt.bar(br6, t6, color='r', width=barWidth,
        edgecolor='grey')
plt.bar(br7, t7, color='r', width=barWidth,
        edgecolor='grey')
plt.bar(br8, t8, color='r', width=barWidth,
        edgecolor='grey')
plt.bar(br9, t9, color='r', width=barWidth,
        edgecolor='grey')
plt.bar(br10, t10, color='r', width=barWidth,
        edgecolor='grey')

# Adding Xticks
plt.ylabel(r'granularity level $k$', fontsize=24)
plt.yticks([i for i in range(3,7)],
           [r'3',r'4',r'5',r'6'], fontsize=24)
plt.xticks([r + 4*barWidth for r in range(len(br10))],
           [r'CNF', r'ARGMAX', r'CDM', r'ground truth'], fontsize=24)

plt.tight_layout()
plt.savefig('rank.pdf')
