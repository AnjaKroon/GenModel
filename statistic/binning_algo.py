
# for this, we already have the samples, but they are not binned
from sampling.discrete import prob_dict_to_array
from statistic.binned import p_to_bp_algo, p_to_bp_random, p_to_bp_with_index


def binning_on_samples(binning_algo, all_trials_q_dict, ground_truth_p_dict, U, B):
    list_binned = []
    for q_dict in all_trials_q_dict:
        if binning_algo == 'random':
            # first, we do the binning randomly, obtain the new binned distribution
            binnned_p_hist, mapping_from_index_to_bin = p_to_bp_random(
                ground_truth_p_dict, U, B)
            binnned_q_hist = p_to_bp_with_index(
                q_dict, U, B, mapping_from_index_to_bin)
        else:
            binnned_p_hist, binnned_q_hist = p_to_bp_algo(
                ground_truth_p_dict, q_dict,  U, B)
        # here its ok to transform the distirbution to an array because the space is supppose to be very small
        binnned_p_array = prob_dict_to_array(binnned_p_hist, B)
        binnned_q_array = prob_dict_to_array(binnned_q_hist, B)

        list_binned.append({'p': binnned_p_array, 'q': binnned_q_array})
    return list_binned


# for this, we already have the samples, but they are not binned
def perform_binning_and_compute_stats(all_trials_q_dict, ground_truth_p_dict, U, B, stat_func):
    list_stat = []
    list_binned_algo = binning_on_samples(
        'algo', all_trials_q_dict, ground_truth_p_dict, U, B)
    list_binned_random = binning_on_samples(
        'random', all_trials_q_dict, ground_truth_p_dict, U, B)
    for i in range(len(all_trials_q_dict)):         #added range
        binned_algo = list_binned_algo[i]
        B_random = stat_func(binned_algo['p'], binned_algo['q'])
        binned_random = list_binned_random[i]
        B_algo = stat_func(binned_random['p'], binned_random['q'])
        list_stat.append({'B_random': B_random, 'B_algo': B_algo})
    return list_stat
