
import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.2
fig = plt.subplots(figsize =(12, 8))
 
# set height of bar
B3 = [0, 100, 100, 100]
B4 = [0, 30, 100, 100]
B5 = [0, 0, 0, 100]
B6 = [0, 0, 0, 100]
 
# Set position of bar on X axis
br1 = np.arange(len(B3))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
 
# Make the plot
plt.bar(br1, B3, color ='r', width = barWidth,
        edgecolor ='grey', label ='k=3')
plt.bar(br2, B4, color ='g', width = barWidth,
        edgecolor ='grey', label ='k=4')
plt.bar(br3, B5, color ='b', width = barWidth,
        edgecolor ='grey', label ='k=5')
plt.bar(br4, B6, color ='k', width = barWidth,
        edgecolor ='grey', label ='k=6')
import matplotlib.font_manager as font_manager
# Adding Xticks
plt.xlabel('Generative models', fontsize = 18)
plt.ylabel('Percentage of trials that passed', fontsize = 18)
plt.xticks([r + barWidth for r in range(len(B3))],
        ['CNF', 'argmax', 'CDM', 'ground truth'],fontsize=18)
font = font_manager.FontProperties(style='normal', size=18)
plt.legend(prop=font)
#plt.show()
plt.savefig('rank.pdf')