import os
import pickle as pk

def read_pickle_file(filename):
    file_name = str(filename)
    path = './S_6_K_6/CNF/14_08_2022__10_54/figure' # Just going to try a single one now
    # later on should be over all .pk files
    file_path = os.path.join(path, file_name)
    with open(file_path, 'rb') as f:
        samples = pk.load(f)
    # print(samples)
    return samples

if __name__ == '__main__':
    read_pickle_file('100sample.pk')