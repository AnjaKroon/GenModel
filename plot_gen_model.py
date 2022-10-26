from sampling.loading_samples import load_generative_model_samples
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from distutils.spawn import find_executable
if find_executable('latex'):
    matplotlib.rcParams['text.usetex'] = True


def compute_dtv(gt, dist_comp):
    # have an array that represents the y values on the empirical pmf
    # have some sort of ground truth function -- unsure how it will come in
    # have an array that represents the ground truth (stair function) that can be used to compare value by value
    # compute for each value in the empirical pmf, empirical pmf[value] - ground truth[value]
    # sum the calculated distances together over the entire space of consideration
    # return the dtv for that specific value
    tot_diff = 0
    for i in range(len(gt)):
        diff = (gt[i] - dist_comp[i])  # **2
        diff = abs(diff)
        tot_diff = tot_diff + diff

    for i in range(len(gt), len(dist_comp)):
        tot_diff += dist_comp[i]

    tot_diff = tot_diff/2

    # print(tot_diff)
    return tot_diff  # aka dtv


if __name__ == '__main__':
    dict_of_samples, ground_truth_dict = load_generative_model_samples(
        num_file=1)
    CDM_empirical_dict = dict_of_samples['cdm'][0]
    argmax_dict = dict_of_samples['argmax'][0]
    CNF_empirical_dict = dict_of_samples['cnf'][0]

    # samples_dict = {'argmax': argmax_samples, 'cdm': cdm_samples,
    #                 'cnf': cnf_samples, 'ground truth': ground_truth_samples_list}
    # return samples_dict, ground_truth_dict

    # making the ground truth dictionary y for compute dtv
    x_gt = list(ground_truth_dict.keys())
    x_sort_arg_gt = np.argsort(x_gt)
    y_gt = list(ground_truth_dict.values())
    x_gt = [x_gt[i] for i in x_sort_arg_gt]
    y_gt = [y_gt[i] for i in x_sort_arg_gt]

    def sorting_plot(holder_dict, label, color):
        x = list(holder_dict.keys())
        x_sort_arg = np.argsort(x)
        y = list(holder_dict.values())
        x = [x[i] for i in x_sort_arg]
        y = [y[i] for i in x_sort_arg]
        y_new = []
        max_index = np.where(np.array(x) < 360)[0].shape[0]
        max_index_end = np.where(np.array(x) < 720)[0].shape[0]
        # indicator function, need to figure out how many elements in x beyond 720
        end_end = np.where(np.array(x) < len(x))[0].shape[0]
        print("end end", end_end)  # how many elements are beyond 720

        y_left = y[0:max_index]
        y_left.sort(reverse=True)
        y_left = y_left + [0 for _ in range(360-max_index)]

        y_right = y[max_index:max_index_end]
        y_right.sort(reverse=True)
        y_right = y_right + [0 for _ in range(720-max_index_end)]

        print("len y", len(y))
        y_zero = y[max_index_end: max_index_end+end_end]
        y_zero.sort(reverse=True)
        # this is how many x units you need to add
        print("len y zero", len(y_zero))

        for item in y_left:
            y_new.append(item)
        for thing in y_right:
            y_new.append(thing)
        for zero in y_zero:
            y_new.append(zero)
        # send y new up to compute dtv

        print("lengths", len(y_gt), len(y_new))
        loc_dtv = compute_dtv(y_gt, y_new)
        label = label + ": d_tv = " + "{:2.4f}".format(loc_dtv)
        plt.plot(list(range(len(y_new))), y_new, color=color, label=label)
    # plt.plot(x[0:720], y[0:720], color='y', label='CDM', linewidth=0.35)
    sorting_plot(CDM_empirical_dict, label='CDM', color='y')
    sorting_plot(argmax_dict, label='ARGMAX', color='b')
    sorting_plot(CNF_empirical_dict, label='CNF', color='c')

    x = list(ground_truth_dict.keys())
    x_sort_arg = np.argsort(x)
    y = list(ground_truth_dict.values())
    x = [x[i] for i in x_sort_arg]
    y = [y[i] for i in x_sort_arg]
    # print(x) # this has all of them
    # print(y) # this has all of them and is in order

    plt.plot(x, y, color='r', label='Ground Truth Distribution')

    # plt.title('100samples.pk')
    plt.ylabel("Empirical pmf")
    plt.legend()
    plt.savefig('emp_gen.pdf')
    plt.close()

    # U posU ratio and S are parameters that will define the stair function
    # stair_histo = make_stair_prob(U, posU, ratio, S)
    # print(np.sum(list(stair_histo.values())))
    # print(stair_histo)
