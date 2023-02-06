
# for this, we already have the samples, but they are not binned
from re import I
from sampling.discrete import prob_dict_to_array
from statistic.binned import p_to_bp_algo


def binning_on_samples(consolidated_samples, trials, ground_truth_p_dict, B, pmf_q):
    list_binned = []
    for trial in range(trials):
        result_dict = p_to_bp_algo(
            ground_truth_p_dict, consolidated_samples,  B, seed=trial, pmf_q=pmf_q)
        # here its ok to transform the distirbution to an array because the space is supppose to be very small

        binnned_p_hist = result_dict['new_histo_p']
        binnned_q_hist = result_dict['new_emp_histo_q']
        binnned_p_array = prob_dict_to_array(binnned_p_hist, B)
        binnned_q_array = prob_dict_to_array(binnned_q_hist, B)
        binned_dict = {'p': binnned_p_array, 'q': binnned_q_array}
        if pmf_q is not None:
            new_histo_q = result_dict['new_histo_q']
            binnned_true_q_array = prob_dict_to_array(new_histo_q, B)
            binned_dict['q_true'] = binnned_true_q_array

        list_binned.append(binned_dict)
    return list_binned
