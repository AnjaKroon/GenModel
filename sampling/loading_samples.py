
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
    for model in list_models:
        model_path = os.path.join(base_path, model)
        date_model = os.listdir(model_path)[0]
        model_path = os.path.join(model_path, date_model)
        model_path = os.path.join(model_path, 'figure')
        for file_name_pickle in pickle_files:
            samples_from_file = read_pickle_file(file_name_pickle, model_path)
            GCDM_empirical_dict = samples_to_histo(samples_from_file)

        # transformer_samples_from_file = read_pickle_file(
        #     file_name_pickle, './S_'+str(power_base)+'_K_'+str(power_base)+'/transformer/2'+str(power_base)+'_09_2022__15_14/figure')
        # transformer_empirical_dict = samples_to_histo(
        #     transformer_samples_from_file)

        arg_max_samples_from_file = read_pickle_file(
            file_name_pickle, './S_'+str(power_base)+'_K_'+str(power_base)+'/argmaxAR/13_08_2022__23_08/figure')
        arg_max_empirical_dict = samples_to_histo(
            arg_max_samples_from_file)

        CDM_samples_from_file = read_pickle_file(
            file_name_pickle, './S_'+str(power_base)+'_K_'+str(power_base)+'/CDM/07_08_2022__11_49/figure')
        CDM_empirical_dict = samples_to_histo(CDM_samples_from_file)

        CNF_samples_from_file = read_pickle_file(
            file_name_pickle, './S_'+str(power_base)+'_K_'+str(power_base)+'/CNF/14_08_2022__10_54/figure')
        CNF_empirical_dict = samples_to_histo(CNF_samples_from_file)

        arg_max_empirical_dict = convert_key_sequence_to_int(
            arg_max_empirical_dict, KEY_CONVERTING_DICT)
        CDM_empirical_dict = convert_key_sequence_to_int(
            CDM_empirical_dict, KEY_CONVERTING_DICT)
        CNF_empirical_dict = convert_key_sequence_to_int(
            CNF_empirical_dict, KEY_CONVERTING_DICT)
        # transformer_empirical_dict = convert_key_sequence_to_int(
        #     transformer_empirical_dict, KEY_CONVERTING_DICT)

        GCDM_empirical_dict = convert_key_sequence_to_int(
            GCDM_empirical_dict, KEY_CONVERTING_DICT)

        # append here and then repeat for another pickle file
        argmax_samples.append(arg_max_empirical_dict)
        cdm_samples.append(CDM_empirical_dict)
        cnf_samples.append(CNF_empirical_dict)
        gcdm_samples.append(GCDM_empirical_dict)
        # transformer_samples.append(transformer_empirical_dict)
    samples_dict = {'argmax': argmax_samples, 'cdm': cdm_samples,
                    'cnf': cnf_samples, 'gcdm': gcdm_samples,  'ground truth': ground_truth_samples_list}
    return samples_dict, ground_truth_dict
