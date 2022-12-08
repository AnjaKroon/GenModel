
import math
import os
from file_helper import read_pickle_file
from sampling.stair import convert_key_sequence_to_int, make_stair_prob, samples_to_histo
from statistic.generate_statistics import generate_samples_scalable


def load_generative_model_samples(power_base, num_files=10, m=10000):
    U = power_base**power_base
    S = 2
    ratio = 3
    ground_truth_dict = make_stair_prob(
        U, posU=(math.factorial(power_base)/U), ratio=ratio,  S=S)
    ground_truth_samples_list = generate_samples_scalable(
        ground_truth_dict, 10, U, m, tempered=False, e=0, b=100)

    pickle_files = ['0sample.pk', '1sample.pk', '2sample.pk', '3sample.pk',
                    '4sample.pk', '5sample.pk', '6sample.pk', '7sample.pk', '8sample.pk', '9sample.pk']
    pickle_files = pickle_files[:num_files]
    base_path = 'S_%d_K_%d' % (power_base, power_base)
    list_models = os.listdir(base_path)
    samples_dict = {}
    for model in list_models:
        samples = []
        model_path = os.path.join(base_path, model)
        date_model = os.listdir(model_path)[0]
        model_path = os.path.join(model_path, date_model)
        model_path = os.path.join(model_path, 'figure')
        for file_name_pickle in pickle_files:
            samples_from_file = read_pickle_file(file_name_pickle, model_path)
            empirical_dict = samples_to_histo(samples_from_file)
            empirical_dict = convert_key_sequence_to_int(empirical_dict)
            samples.append(empirical_dict)
        samples_dict[model] = samples
    samples_dict['ground truth'] = ground_truth_samples_list
    return samples_dict, ground_truth_dict
