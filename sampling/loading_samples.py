
import math
import os
from file_helper import read_pickle_file
from sampling.stair import convert_key_sequence_to_int, make_stair_prob, samples_to_histo
from statistic.generate_statistics import generate_samples_scalable
import numpy as np
import math


def get_closest_smaller_perm(sequence, base):
    x_dict = {}
    reminding = list(range(base))
    closest_smaller_permuation = []
    for x in sequence:
        if x in x_dict:
            usable_x = [r < x for r in reminding]
            index = int(np.sum(usable_x))-1
            if index < 0:  # go back
                for i in list(reversed(range(len(closest_smaller_permuation)))):
                    xs = closest_smaller_permuation[i]
                    usable_x = [r < xs for r in reminding]
                    index = int(np.sum(usable_x))-1
                    if index >= 0:  # replace the end
                        closest_smaller_permuation = closest_smaller_permuation[:i]
                        closest_smaller_permuation.append(reminding[index])
                        for x in reversed(range(base)):
                            if x not in closest_smaller_permuation:
                                closest_smaller_permuation.append(x)
                        return closest_smaller_permuation
                    else:
                        reminding.append(xs)
                        reminding.sort()
                return None
            else:
                closest_smaller_permuation.append(reminding[index])
                if reminding[index] in reminding:
                    reminding.remove(reminding[index])
                closest_smaller_permuation = closest_smaller_permuation + \
                    list(reversed(reminding))
                return closest_smaller_permuation
        else:
            closest_smaller_permuation.append(x)
            x_dict[x] = 1
            reminding.remove(x)
    return closest_smaller_permuation


def count_permutation_before(sequence, base):
    # find closest smallest permutation before
    closest_smaller_permuation = get_closest_smaller_perm(sequence, base)
    if closest_smaller_permuation is not None:
        count_permutation_before = get_permuation_rank(
            closest_smaller_permuation, base) + 1
        return count_permutation_before
    else:
        return 0


def likely_with_start(b_sequence, base):
    remainding = list(range(base))
    for s in b_sequence:
        remainding.remove(s)
    can_end = np.sum([r > b_sequence[0] for r in remainding])
    likely_with_start = can_end * math.factorial(base-len(b_sequence)-1)
    return likely_with_start


def to_permutation(b_sequence, base):
    d = {}
    s = []
    for b in b_sequence:
        if b not in d:
            s.append(b)
            d[b] = 1
        else:
            while b in d:
                b = b+1

            if b < base and b >= 0:
                d[b] = 1
                s.append(b)
            else:
                return None

    return s


def get_type(comb):
    choices = {}
    for x in comb:
        if x not in choices:
            choices[x] = 1
        else:
            return 'not a permutation'
    if comb[0] < comb[-1]:
        return 'likely'
    else:
        return 'rare'


def count_likely_before(sequence, base):
    start = sequence[0]
    # all likely that starts with a lower number will be before.
    # if start = 2, 0123, 0132, .... then 1032, 1302, ...
    num_likely_before = [(base-i-1)*math.factorial(base-2)
                         for i in range(start)]
    num_likely_before = int(np.sum(num_likely_before))
    # then all likely that starts with same number
    num_likely_at = (base-start-1) * math.factorial(base-2)

    for i in range(1, base-1):
        begining_sequence = sequence[:i]
        begining_s = to_permutation(begining_sequence, base)
        if begining_s is not None:
            for b in range(sequence[i]+1, base):
                b_sequence = begining_s + [b]
                if get_type(b_sequence) != 'not a permutation':
                    inaccessible_likely = likely_with_start(b_sequence, base)
                    num_likely_at = num_likely_at-inaccessible_likely
    num_likely_at = num_likely_at-1  # remove the sequence itself
    return num_likely_before + num_likely_at


def get_permuation_rank(sequence, base):
    seq_id = 0
    nums_left = list(range(base))
    for s, x in enumerate(sequence):
        offset = nums_left.index(x)*math.factorial(base-s-1)
        nums_left.remove(x)
        nums_left.sort()
        seq_id += offset
    return seq_id


def get_sequence_rank(sequence, base):
    seq_id = 0
    # count in base
    for s, x in enumerate(sequence):
        seq_id += base**(base-s-1) * x
    return seq_id


def sequence_to_id(sequence, base):
    seq_id = 0

    # print(nums_left)
    type_seq = get_type(sequence)
    if type_seq == 'likely' or type_seq == 'rare':

        if type_seq == 'rare':
            seq_id = get_permuation_rank(
                sequence, base) - count_likely_before(sequence, base) - 1
            seq_id = seq_id+int(math.factorial(base)/2)
        else:
            seq_id = count_likely_before(sequence, base)

    else:
        seq_id = get_sequence_rank(sequence, base)
        seq_id += math.factorial(base)  # add all permutation at the beginnig
        # remove the already accounted for permutation.
        seq_id = seq_id-count_permutation_before(sequence, base)

    return seq_id


def load_generative_model_samples(power_base, num_files=10, m=10000):
    U = power_base**power_base
    S = 2
    ratio = 3
    ground_truth_dict = make_stair_prob(
        U, posU=(math.factorial(power_base)/U), ratio=ratio,  S=S)
    ground_truth_samples_list = generate_samples_scalable(
        ground_truth_dict, num_files, U, m, tempered=False, e=0, b=100)['all_trials_emp']
    zero_space = 0
    for key, val in ground_truth_samples_list[0].items():
        if key > math.factorial(power_base):
            zero_space +=1
    # pickle_files = ['0sample.pk', '1sample.pk', '2sample.pk', '3sample.pk',
    #                 '4sample.pk', '5sample.pk', '6sample.pk', '7sample.pk', '8sample.pk', '9sample.pk']
    
    #pickle_files = pickle_files[:num_files]
    pickle_files = ['100sample.pk']
    base_path = 'S_%d_K_%d' % (power_base, power_base)
    list_models = os.listdir(base_path)
    samples_dict = {}
    
    for model in list_models:
        samples = []
        model_path = os.path.join(base_path, model)
        list_models_date = os.listdir(model_path)
        list_models_date.sort()
        date_model = list_models_date[-1]
        model_path = os.path.join(model_path, date_model)
        model_path = os.path.join(model_path, 'figure')
        for file_name_pickle in pickle_files:
            samples_from_file = read_pickle_file(file_name_pickle, model_path)
            empirical_dict = samples_to_histo(samples_from_file)
            empirical_dict = convert_key_sequence_to_int(
                power_base, empirical_dict, sequence_to_id)
            samples.append(empirical_dict)
        samples_dict[model] = samples
    samples_dict['ground truth'] = ground_truth_samples_list
    return samples_dict, ground_truth_dict
