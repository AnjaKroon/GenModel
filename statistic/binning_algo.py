
# for this, we already have the samples, but they are not binned
from sampling.discrete import prob_dict_to_array
from statistic.binned import p_to_bp_algo


def binning_on_samples(consolidated_samples, trials, ground_truth_p_dict, U, B):
    list_binned = []
    for trial in range(trials):
        binnned_p_hist, binnned_q_hist = p_to_bp_algo(
                ground_truth_p_dict, consolidated_samples,  B, seed=trial)
        # here its ok to transform the distirbution to an array because the space is supppose to be very small
        binnned_p_array = prob_dict_to_array(binnned_p_hist, B)
        binnned_q_array = prob_dict_to_array(binnned_q_hist, B)

        list_binned.append({'p': binnned_p_array, 'q': binnned_q_array})
    return list_binned

