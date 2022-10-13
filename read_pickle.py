import os
import pickle as pk

def read_pickle_file(filename, p):
    file_name = str(filename)
    path = str(p) # argmaxAR case -- pretty graph
    # path = './S_6_K_6/CDM/07_08_2022__11_49/figure' # CDM option 1 -- not bad graph
    # later on should be over all .pk files
    file_path = os.path.join(path, file_name)
    with open(file_path, 'rb') as f:
        samples = pk.load(f)
    # print(samples)
    return samples

if __name__ == '__main__':
    read_pickle_file('100sample.pk')