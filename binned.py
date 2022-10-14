#from discrete import makeUniProbArr, errFunct, genValArr, sampleSpecificProbDist
#from gen_S import empirical_dist, genSstat
#from plot_utils import plot_S_stat
#from sampling.poisson import poisson_empirical_dist
from decimal import ROUND_DOWN
from re import L
import sys
import numpy as np
import random
import math

# return the dict of a binned pmf, with predined binning mapping_from_index_to_bin


def p_to_bp_with_index(histo_p, U, B, mapping_from_index_to_bin):
    new_histo = {}
    for index, val in histo_p.items():
        bin = mapping_from_index_to_bin[index]
        if bin not in new_histo:
            new_histo[bin] = val
        else:
            new_histo[bin] += val
    return new_histo

# assign a bin at randoms to each element


def p_to_bp_random(histo_p, U, B):
    amount_per_bin = math.floor(U/B)  # 3
    amount_final_bin = int(amount_per_bin + (U % B))  # 4

    # shuffle the binning
    mapping_from_index_to_bin = {}
    mapping_bin_to_index = {}
    shuffled_U = list(range(U))
    random.shuffle(shuffled_U)
    for i in range(B):
        mapping_bin_to_index[i] = []
        size_bin = amount_per_bin
        if i == B-1:
            size_bin = amount_final_bin
        for j in range(size_bin):
            index = i * amount_per_bin + j
            shuffled_index = shuffled_U[index]
            mapping_from_index_to_bin[shuffled_index] = i
            mapping_bin_to_index[i].append(shuffled_index)
    new_histo = {}
    for bin_index, all_index in mapping_bin_to_index.items():
        new_probability_for_bin = 0
        for j in all_index:
            # if j not in histo_p, we assume that histo_p[j] = 0, so we can skip
            if j in histo_p:
                new_probability_for_bin = new_probability_for_bin + histo_p[j]
        new_histo[bin_index] = new_probability_for_bin
    return new_histo, mapping_from_index_to_bin


def find_flat_regions(ground_truth_p_dict):
    # todo , just return as is
    flat_regions = {}
    for key, p_val in ground_truth_p_dict.items():
        if p_val in flat_regions:
            flat_regions[p_val].append(key)
        else:
            flat_regions[p_val] = [key]
    # sort all flat regions
    for p_val in flat_regions.keys():
        flat_regions[p_val].sort()
    return flat_regions

# use the algo to assign a bin to each algo


def split(list_a, chunk_size):

    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]


def p_to_bp_algo(ground_truth_p_dict, q_dict,  U, B):

    predefined_bins_with_error = {}
    # be default, there is always the null space as well
    flat_regions = find_flat_regions(ground_truth_p_dict)
    num_pos_flat_regions = 0
    zero_error_regions = 0
    region_index = 0
    for s_flat, indices in flat_regions.items():
        list_errors_index = []
        for index in indices:
            if index in q_dict:
                q_x = q_dict[index]
                p_x = s_flat
                error = p_x - q_x
                list_errors_index.append((error, index))
            else:
                q_x = 0
                p_x = s_flat
                error = p_x - q_x
                list_errors_index.append((error, index))

        # we separate the positive from the negative error to find where the bin split would be
        cumul_pos_error = 0
        cumul_neg_error = 0
        # these lists will form the bins.
        indices_pos_bin = []  # this could stay empty
        indices_neg_bin = []  # this could stay empty
        for error, index in list_errors_index:
            if error > 0:
                indices_pos_bin.append(index)
                cumul_pos_error += error
            else:
                indices_neg_bin.append(index)
                cumul_neg_error += -error
        # the error that will be lost if we dont cut this bin
        cut_error = cumul_neg_error + cumul_pos_error - \
            np.abs((cumul_pos_error-cumul_neg_error))
        # now we know how much error is contained in a split
        predefined_bins_with_error[region_index] = {
            'cut_error': cut_error, 'pos_indices': indices_pos_bin, 'neg_indices': indices_neg_bin}
        if cut_error == 0:
            zero_error_regions += 1
        else:
            num_pos_flat_regions += 1  # increment the flat region index
        region_index += 1
    # find which bin have the more potential error if cut
    if zero_error_regions > 1:
        print('Warning, we got zero error regions')
    sorted_s_by_potential_cut_error = sorted(list(predefined_bins_with_error.keys()),
                                             key=lambda x: predefined_bins_with_error[x]['cut_error'])
    sorted_s_by_potential_cut_error.reverse()  # highest to lowest
    num_region_that_can_be_cut = (num_pos_flat_regions-zero_error_regions)
    mapping_bin_to_index = {}
    mapping_from_index_to_bin = {}
    # now we need to cut the predefined bins in the number of wanted bins B
    if B < num_pos_flat_regions+1:  # NOT EXACT SOL
        raise NotImplemented
    # easiest scenario, we just return all flat regions
    elif B == num_pos_flat_regions+zero_error_regions+1:
        for region_to_left_uncut in range(num_pos_flat_regions+zero_error_regions):
            dict_pos_neg = predefined_bins_with_error[region_to_left_uncut]
            indices_of_the_whole_region = dict_pos_neg['pos_indices'] + \
                dict_pos_neg['neg_indices']
            mapping_bin_to_index[region_to_left_uncut] = indices_of_the_whole_region

    elif B > (num_pos_flat_regions+1) and B <= (2*num_region_that_can_be_cut + 1 + zero_error_regions):
        bin_ind = 0
        how_many_flat_region_we_should_cut = B - num_pos_flat_regions - 1
        list_of_remaining_regions = list(range(num_pos_flat_regions))
        for i in range(how_many_flat_region_we_should_cut):
            # by default, the zero error will be at the end so we can leave them in
            region_to_cut = sorted_s_by_potential_cut_error[i]
            list_of_remaining_regions.remove(region_to_cut)
            mapping_bin_to_index[bin_ind] = predefined_bins_with_error[region_to_cut]['pos_indices']
            bin_ind += 1
            mapping_bin_to_index[bin_ind] = predefined_bins_with_error[region_to_cut]['neg_indices']
            bin_ind += 1
        for region_to_left_uncut in list_of_remaining_regions:
            dict_pos_neg = predefined_bins_with_error[region_to_left_uncut]
            indices_of_the_whole_region = dict_pos_neg['pos_indices'] + \
                dict_pos_neg['neg_indices']
            mapping_bin_to_index[bin_ind] = indices_of_the_whole_region
            bin_ind += 1
    # B>= 2s, we randomly cut the bins, nothing else to be done. The error will be flat at this point.
    elif num_region_that_can_be_cut > 0:
        num_random_cut = B - 2*num_region_that_can_be_cut - 1 - zero_error_regions
        random.seed(2)

        bin_with_cut = random.choices(
            list(range(num_region_that_can_be_cut*2)), k=num_random_cut)

        cuts_per_bin = {}
        for c in bin_with_cut:
            if c in cuts_per_bin:
                cuts_per_bin[c] += 1
            else:
                cuts_per_bin[c] = 1
        bin_ind = 0
        list_of_remaining_regions = list(range(num_pos_flat_regions))
        for i in range(num_region_that_can_be_cut):
            region_to_cut = sorted_s_by_potential_cut_error[i]
            list_of_remaining_regions.remove(region_to_cut)
            pos_indices_region_i = predefined_bins_with_error[region_to_cut]['pos_indices']
            pos_ind = i*2
            cuts_per_this_region = 0
            if pos_ind in cuts_per_bin:
                cuts_per_this_region = cuts_per_bin[pos_ind]

            chunks_of_pos = split(pos_indices_region_i, math.ceil(
                len(pos_indices_region_i)/(1+cuts_per_this_region)))
            for chunk_indices in chunks_of_pos:
                mapping_bin_to_index[bin_ind] = chunk_indices
                bin_ind += 1

            neg_indices_region_i = predefined_bins_with_error[region_to_cut]['neg_indices']
            neg_ind = i*2+1
            cuts_per_this_region = 0
            if neg_ind in cuts_per_bin:
                cuts_per_this_region = cuts_per_bin[neg_ind]
            chunks_of_neg = split(neg_indices_region_i, math.ceil(
                len(neg_indices_region_i)/(1+cuts_per_this_region)))
            for chunk_indices in chunks_of_neg:
                mapping_bin_to_index[bin_ind] = chunk_indices
                bin_ind += 1
        for region_to_left_uncut in list_of_remaining_regions:
            dict_pos_neg = predefined_bins_with_error[region_to_left_uncut]
            indices_of_the_whole_region = dict_pos_neg['pos_indices'] + \
                dict_pos_neg['neg_indices']
            mapping_bin_to_index[bin_ind] = indices_of_the_whole_region
            bin_ind += 1
    else:
        num_random_cut = B - 1 - zero_error_regions
        random.seed(2)
        bin_with_cut = random.choices(list(range(zero_error_regions)), k=num_random_cut)

        cuts_per_bin = {}
        for c in bin_with_cut:
            if c in cuts_per_bin:
                cuts_per_bin[c] += 1
            else:
                cuts_per_bin[c] = 1
        bin_ind = 0

        for region_to_cut in list(range(zero_error_regions)):
            if region_to_cut in cuts_per_bin:
                num_cuts = cuts_per_bin[region_to_cut]
                dict_pos_neg = predefined_bins_with_error[region_to_cut]
                indices_of_the_whole_region = dict_pos_neg['pos_indices'] + \
                    dict_pos_neg['neg_indices']
                chunks = split(indices_of_the_whole_region, math.ceil(
                    len(indices_of_the_whole_region)/(1+num_cuts)))
                for chunk_indices in chunks:
                    mapping_bin_to_index[bin_ind] = chunk_indices
                    bin_ind += 1
            else:
                dict_pos_neg = predefined_bins_with_error[region_to_cut]
                indices_of_the_whole_region = dict_pos_neg['pos_indices'] + \
                    dict_pos_neg['neg_indices']
                mapping_bin_to_index[bin_ind] = indices_of_the_whole_region
                bin_ind += 1

    new_histo_p = {}
    for bin_index, all_index in mapping_bin_to_index.items():
        new_probability_for_bin = 0
        for j in all_index:
            if j in ground_truth_p_dict:
                new_probability_for_bin = new_probability_for_bin + \
                    ground_truth_p_dict[j]
        new_histo_p[bin_index] = new_probability_for_bin

    new_histo_q = {}

    for bin_index, all_index in mapping_bin_to_index.items():
        new_probability_for_bin = 0
        for j in all_index:
            if j in q_dict:
                new_probability_for_bin = new_probability_for_bin + q_dict[j]
        new_histo_q[bin_index] = new_probability_for_bin

    # add the zero bin
    new_histo_q[B] = 1 - np.sum(list(new_histo_q.values()))
    new_histo_p[B] = 0
    return new_histo_p, new_histo_q


def transform_samples(b_p, histo_p, p_samples, U, B):
    # define subdivisions
    print("size of p_samples is ", len(p_samples))
    amount_per_bin = math.floor(len(p_samples)/B)
    amount_final_bin = amount_per_bin + (len(p_samples) % B)
    # bins = B
    new_samples = []
    # for item in histo_p.items():
    # if sample is from 0 to 33, add element [1] to new_samples
    # if sample is from 34 to 66, add element [2] to new_samples
    # if sample is from 67 to 100, add element [3] to new_samples
    for i in range(1, B+1):
        if i != B:
            for amt in range(amount_per_bin):
                new_samples.append(i)
        if i == B:
            for amt in range(amount_final_bin):
                new_samples.append(i)

    print(new_samples)

    # for i amount of bins
    # # i = 1 lets say
    # if not last bin, let's add numbers, for the amount_per_bin
    # if last bin, let's add numbers, for the amount_final_bin
    return new_samples


if __name__ == '__main__':
    sample_histo = {1: 0.15, 2: 0.15, 3: 0.1, 4: 0.05,
                    5: 0.1, 6: 0.1, 7: 0.05, 8: 0.1, 9: 0.1, 10: 0.1}
    sample_b = {1: 0.33, 2: 0.33, 3: 0.33}
    p_samples = [1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 4, 5, 6, 7, 8, 9, 10]

    U = 10
    B = 3

    b_p = p_to_bp_random(sample_histo, U, B)  # works
    print(b_p)
    print('probability sum to : ', sum(b_p.values()))
    b_out = transform_samples(
        b_p, sample_histo, p_samples, U, B)  # returns new samples

    # the question is how to get the relevant things in
    # need some sort of histo for the transform_samples -- there should be an array of the poissonized samples
