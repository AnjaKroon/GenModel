import os
import pickle as pk 

file_name = '100sample.pk'
path = './S_6_K_6/CNF/14_08_2022__10_54/figure/100sample.pk ' # Just going to try a single one now
# later on should be over all .pk files
file_path = os.path.join(path, file_name)
with open(file_path, 'rb') as f:
    samples = pk.load(f)
print(samples)