import numpy as np
import random
import math
from sampling.discrete import find_interval
random.seed(2)
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

# the historgram could be


def get_probability_at_element(j, histogram):
    is_optimized = type(list(histogram.values())[0]) is dict

    if is_optimized:
        intervals = [val['interval'] for _, val in histogram.items()]
        interval_index = find_interval(j, intervals)
        if interval_index in histogram:
            return histogram[interval_index]['p']
    else:
        if j in histogram:
            return histogram[j]
    return 0  # if the index doesn't appear, we assume the prob is 0

# assign a bin at randoms to each element


def p_to_bp_random(ground_truth_p_dict, U, B):

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
            new_probability_for_bin = new_probability_for_bin + \
                get_probability_at_element(j, ground_truth_p_dict)
        new_histo[bin_index] = new_probability_for_bin
    return new_histo, mapping_from_index_to_bin


def find_flat_regions(ground_truth_p_dict, is_optimized):
    if is_optimized:
        flat_regions = {}
        for key, val in ground_truth_p_dict.items():
            p_val = val['p']
            interval = val['interval']
            flat_regions[p_val] = range(interval[0], interval[1])

        return flat_regions
    else:
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


def collecting_error(regions, q_dict):
    predefined_bins_with_error = {}
    regions_that_should_be_cut = 0  # regions that have positive and negative p_x - q_x
    regions_that_should_not_be_cut = 0  # regions that only have pos. or neg p_x - q_x
    s = 0
    for s_flat, indices in regions.items():
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
        predefined_bins_with_error[s] = {
            'cut_error': cut_error, 'pos_indices': indices_pos_bin, 'neg_indices': indices_neg_bin}
        if cut_error == 0:
            regions_that_should_not_be_cut += 1
        else:
            regions_that_should_be_cut += 1  # increment the flat region index
        s += 1
    return predefined_bins_with_error, regions_that_should_be_cut, regions_that_should_not_be_cut


def p_to_bp_algo(ground_truth_p_dict, q_dict,  U, B):
    is_optimized = type(list(ground_truth_p_dict.values())[0]) is dict

    # 1 : find the S partitioning. By default, the zero space is "assumed" but not computed here.
    regions = find_flat_regions(ground_truth_p_dict, is_optimized)

    # 2 : Find each optimal split in each S regions. If the error is all pos or neg, there is no optimal split.

    # find B* for k = 2s
    # predefined_bins_with_error = {0 : {'cut_error':0.xx, 'pos_indices': [2,3,6..],'neg_indices': [1,4,7..]},
    #                               1 : {...}}
    predefined_bins_with_error, regions_that_could_be_cut, regions_that_should_not_be_cut = collecting_error(
        regions, q_dict)

    if regions_that_should_not_be_cut > 1:  # some regions dont have an optimal split
        print('Warning, we got zero error regions')

    # 3 : Sort the S regions by max to min to give B* for a specific k.
    sorted_s_by_potential_cut_error = sorted(list(predefined_bins_with_error.keys()),
                                             key=lambda x: predefined_bins_with_error[x]['cut_error'])
    sorted_s_by_potential_cut_error.reverse()  # highest to lowest

    S = regions_that_could_be_cut + regions_that_should_not_be_cut + 1
    max_B = 2*regions_that_could_be_cut + 1 + regions_that_should_not_be_cut
    # 4 : Return B* : mapping_bin_to_index is {index_bin: [all x], ...}
    mapping_bin_to_index = {}
    if B < S:  # NOT EXACT SOL
        raise NotImplemented
    # easiest scenario k=S, we just return all flat regions without cutting
    elif B == S:

        # S-1 because the zero region is implied
        for bin_ind, region_indices in enumerate(regions.values()):
            mapping_bin_to_index[bin_ind] = region_indices

    elif B > S:
        bin_ind = 0
        if B <= max_B:
            num_regions_have_to_cut = math.floor(B-S)
        else:  # if we have more cuts than predefined regions (B > max_B), we start randomly cutting
            num_regions_have_to_cut = regions_that_could_be_cut
        list_of_remaining_regions = list(range(regions_that_could_be_cut))
        for i in range(num_regions_have_to_cut):
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
        if B > max_B:  # if we have more cuts than predefined regions (B > max_B), we start randomly cutting
            num_random_cut = B - max_B
            num_random_cute_candidate = len(mapping_bin_to_index)
            bin_with_random_cut = random.choices(
                list(range(num_random_cute_candidate)), k=num_random_cut)
            random_cuts_per_bin = np.histogram(
                bin_with_random_cut, bins=list(range(num_random_cut+1)))[0]
            
            for bin_id_to_cut, num_random_cuts in enumerate(random_cuts_per_bin):
                indices_to_split = mapping_bin_to_index[bin_id_to_cut]
                chunk_size = int(len(indices_to_split)/(num_random_cuts+1))
                # first, we replace the bin by a small chunk of the indices:
                mapping_bin_to_index[bin_id_to_cut] = indices_to_split[:chunk_size] 
                # then we create news bins with all the chunks
                chunk_id = 1
                for chunk_id in range(1, num_random_cuts):
                    mapping_bin_to_index[bin_ind] = indices_to_split[chunk_id*chunk_size:(chunk_id+1)*chunk_size]
                    bin_ind += 1
                mapping_bin_to_index[bin_ind] = indices_to_split[chunk_id*chunk_size:]
       

    new_histo_p = {}
    for bin_index, all_index in mapping_bin_to_index.items():
        new_probability_for_bin = 0
        for j in all_index:
            new_probability_for_bin = new_probability_for_bin + \
                get_probability_at_element(j, ground_truth_p_dict)

        new_histo_p[bin_index] = new_probability_for_bin

    new_histo_q = {}

    for bin_index, all_index in mapping_bin_to_index.items():
        new_probability_for_bin = 0
        for j in all_index:
            new_probability_for_bin = new_probability_for_bin + \
                get_probability_at_element(j, q_dict)
        new_histo_q[bin_index] = new_probability_for_bin

    # add the zero bin
    new_histo_q[B-1] = 1 - np.sum(list(new_histo_q.values()))
    new_histo_p[B-1] = 0
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
    ground_truth_p_dict = {0: 0.2, 1: 0.2, 2: 0.2,
                           3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0, 8: 0, 9: 0}
    q_dict = {0: 0.25, 1: 0.25, 2: 0.1,
              3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0, 8: 0, 9: 0}
    U = 10
    B = 3
    p_to_bp_algo(ground_truth_p_dict, q_dict,  U, B)

    # the question is how to get the relevant things in
    # need some sort of histo for the transform_samples -- there should be an array of the poissonized samples
