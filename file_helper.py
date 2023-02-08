import os
import pickle as pk
from os import listdir
from os.path import isfile, join
from statistic.generate_statistics import generate_samples_scalable


def load_all_files(path):

    return [f for f in listdir(path) if isfile(join(path, f))]


def store_for_plotting(data, title):
    file_path = os.path.join('results', title+'.pk')
    with open(file_path, 'wb') as f:  # store the data
        pk.dump(data, f)


def read_pickle_file(filename, p):
    file_name = str(filename)
    path = str(p)  # argmaxAR case -- pretty graph
    # path = './S_6_K_6/CDM/07_08_2022__11_49/figure' # CDM option 1 -- not bad graph
    # later on should be over all .pk files
    file_path = os.path.join(path, file_name)
    with open(file_path, 'rb') as f:
        samples = pk.load(f)
    # print(samples)
    return samples

# this will try to load the data, if the data isn't there,
# it will call the generating_func which returns the data and store it.


def read_file_else_store(file_path, generating_func):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:  # load the data
            data = pk.load(f)
    else:
        print('Couldnt find', file_path, 'generating and storing...')
        data = generating_func()
        with open(file_path, 'wb') as f:  # store the data
            pk.dump(data, f)
    return data

# simple function to quickly generate name of files


def create_prefix_from_list(dict_name):
    list_word = []
    for key, val in dict_name.items():
        try:
            if isinstance(val, int):
                word = str(val)
            else:
                word = "{:.2f}".format(val)
        except Exception:
            word = val
        list_word.append(key+':'+word)
    prefix_srt = '_'.join(list_word)
    return prefix_srt
# this function will either load existing samples or generate new ones


def load_samples(list_of_espilon_q, b, ground_truth_p, splits, U, m, S, ratio, TYPE):
    # obtain the samples
    list_of_samples = []
    list_of_pmf_q = []
    directory_samples_file = 'samples_storing'
    for e in list_of_espilon_q:
        sample_file = create_prefix_from_list(
            {'exp': 'SYNTH'+TYPE, 'U': U, 'm': m, 'splits': splits, 'S': S, 'ratio': ratio, 'b': b, 'e': e}) + '_samples.pk'

        sample_file_path = os.path.join(directory_samples_file, sample_file)

        def generating_samples_func():
            if e == 0:
                samples_and_pmf = generate_samples_scalable(
                    ground_truth_p, splits, U, m, tempered=False, e=0, b=100, TYPE=TYPE)

            else:
                samples_and_pmf = generate_samples_scalable(
                    ground_truth_p, splits, U, m, tempered=True, e=e, b=b, TYPE=TYPE)

            return samples_and_pmf
        # generating_samples_func()
        samples_and_pmf = read_file_else_store(
            sample_file_path, generating_samples_func)
        samples = samples_and_pmf['splits_q_emp']
        pmf_q = samples_and_pmf['q']
        list_of_samples.append(samples)
        list_of_pmf_q.append(pmf_q)

    return list_of_samples, list_of_pmf_q


if __name__ == '__main__':
    read_pickle_file('100sample.pk')
