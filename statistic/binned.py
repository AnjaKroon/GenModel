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



def get_summed_probabilities_of_interval(interval, histogram):
    is_optimized = type(list(histogram.values())[0]) is dict
    start = interval[0]
    end = interval[1]
    if is_optimized:
        probability = get_probability_at_element(start, histogram)
        summed_prob = probability * (end-start)
    else:  # we have to go through all keys
        summed_prob = 0
        for key, val in histogram.items():
            if key >= start and key < end:
                summed_prob+=val

    return summed_prob  # we always only have one prob per region


def p_to_bp_algo(ground_truth_p_dict, q_dict,  U, B):
    is_optimized = type(list(ground_truth_p_dict.values())[0]) is dict

    # 1 : find the S partitioning. By default, the zero space is "assumed" but not computed here.
    regions = find_flat_regions(ground_truth_p_dict, is_optimized)

    # 2 : Find each optimal split in each S regions. If the error is all pos or neg, there is no optimal split.

    S = len(regions) + 1

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
        num_random_cut = math.floor(B-S)

        for bin_ind, region_indices in enumerate(regions.values()):
            mapping_bin_to_index[bin_ind] = region_indices

        num_random_cute_candidate = len(mapping_bin_to_index)
        random_cuts_per_bin = [0 for _ in range(num_random_cute_candidate)]

        random.seed(1)  # this is to keep randomnes
        for _ in range(num_random_cut):
            index = random.randint(0, num_random_cute_candidate-1)
            random_cuts_per_bin[index] += 1
        
        for bin_id_to_cut, num_random_cuts in enumerate(random_cuts_per_bin):
            if num_random_cuts > 0:
                # if num_random_cuts>1:
                # print('problem')
                indices_to_split = mapping_bin_to_index[bin_id_to_cut]
                chunk_size = int(len(indices_to_split)/(num_random_cuts+1))
                # first, we replace the bin by a small chunk of the indices:
                mapping_bin_to_index[bin_id_to_cut] = indices_to_split[:chunk_size]
                # then we create news bins with all the chunks
                chunk_id = 0

                bin_ind += 1
                for chunk_id in range(1, num_random_cuts):
                    mapping_bin_to_index[bin_ind] = indices_to_split[chunk_id *
                                                                     chunk_size: (chunk_id+1)*chunk_size]
                    bin_ind += 1

                mapping_bin_to_index[bin_ind] = indices_to_split[(
                    chunk_id+1)*chunk_size:]

    new_histo_p = {}
    if is_optimized:
        for bin_index, all_index in mapping_bin_to_index.items():
            interval = (all_index[0], all_index[-1])
            new_probability_for_bin = get_summed_probabilities_of_interval(
                interval, ground_truth_p_dict)

            new_histo_p[bin_index] = new_probability_for_bin

        new_histo_q = {}

        for bin_index, all_index in mapping_bin_to_index.items():
            interval = (all_index[0], all_index[-1])
            new_probability_for_bin = get_summed_probabilities_of_interval(
                interval, q_dict)
            new_histo_q[bin_index] = new_probability_for_bin

        # add the zero bin
        new_histo_q[B-1] = 1 - np.sum(list(new_histo_q.values()))
        new_histo_p[B-1] = 0
        return new_histo_p, new_histo_q

    else:

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
