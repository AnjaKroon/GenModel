
from file_helper import read_pickle_file
from sampling.stair import build_ground_truth_dict, convert_key_sequence_to_int, get_converting_dict, samples_to_histo
from statistic.generate_statistics import generate_samples_scalable


def load_generative_model_samples(num_file=10, m=10000):
    U = 6**6
    ground_truth_dict = build_ground_truth_dict()  # same for all
    KEY_CONVERTING_DICT = get_converting_dict()  # same for all
    ground_truth_dict = convert_key_sequence_to_int(
        ground_truth_dict, KEY_CONVERTING_DICT)  # same for all

    ground_truth_samples_list = generate_samples_scalable(
        ground_truth_dict, 10, U, m, tempered=False, e=0, b=100)

    pickle_files = ['0sample.pk', '1sample.pk', '2sample.pk', '3sample.pk',
                    '4sample.pk', '5sample.pk', '6sample.pk', '7sample.pk', '8sample.pk', '9sample.pk']
    pickle_files = pickle_files[:num_file]
    argmax_samples = []
    cdm_samples = []
    cnf_samples = []
    gcdm_samples = []
  #  transformer_samples = []
    for file_name_pickle in pickle_files:
        GCDM_samples_from_file = read_pickle_file(
            file_name_pickle, './S_6_K_6/FCDM/02_09_2022__10_28/figure')
        GCDM_empirical_dict = samples_to_histo(GCDM_samples_from_file)

        # transformer_samples_from_file = read_pickle_file(
        #     file_name_pickle, './S_6_K_6/transformer/26_09_2022__15_14/figure')
        # transformer_empirical_dict = samples_to_histo(
        #     transformer_samples_from_file)

        arg_max_samples_from_file = read_pickle_file(
            file_name_pickle, './S_6_K_6/argmaxAR/13_08_2022__23_08/figure')
        arg_max_empirical_dict = samples_to_histo(
            arg_max_samples_from_file)

        CDM_samples_from_file = read_pickle_file(
            file_name_pickle, './S_6_K_6/CDM/07_08_2022__11_49/figure')
        CDM_empirical_dict = samples_to_histo(CDM_samples_from_file)

        CNF_samples_from_file = read_pickle_file(
            file_name_pickle, './S_6_K_6/CNF/14_08_2022__10_54/figure')
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
