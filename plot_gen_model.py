from sampling.loading_samples import load_generative_model_samples
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dict_of_samples, ground_truth_dict = load_generative_model_samples(
        num_file=1)
    CDM_empirical_dict = dict_of_samples['cdm'][0]
    argmax_dict = dict_of_samples['argmax'][0]
    #ground_truth_dict =  dict_of_samples['ground truth'][0]
    CNF_empirical_dict = dict_of_samples['cnf'][0]
    # CDM_empirical_dict =

    # samples_dict = {'argmax': argmax_samples, 'cdm': cdm_samples,
    #                 'cnf': cnf_samples, 'ground truth': ground_truth_samples_list}
    # return samples_dict, ground_truth_dict
    larger_zero =
    def sorting_plot(holder_dict, label, color):

        x = list(holder_dict.keys())
        x_sort_arg = np.argsort(x)
        y = list(holder_dict.values())
        x = [x[i] for i in x_sort_arg]
        y = [y[i] for i in x_sort_arg]
        y_new = []
        max_index = np.where(np.array(x) < 360)[0].shape[0]
        max_index_end = np.where(np.array(x) < 720)[0].shape[0]
        y_left = y[0:max_index]
        y_left.sort(reverse=True)
        y_left = y_left + [0 for _ in range(360-max_index)]

        y_right = y[max_index:max_index_end]
        y_right.sort(reverse=True)
        y_right = y_right + [0 for _ in range(720-max_index_end)]

        for item in y_left:
            y_new.append(item)
        for thing in y_right:
            y_new.append(thing)

        plt.plot(list(range(720)), y_new, color=color, label=label)
    # plt.plot(x[0:720], y[0:720], color='y', label='CDM', linewidth=0.35)
    sorting_plot(CDM_empirical_dict, label='CDM', color='y')
    sorting_plot(argmax_dict, label='ARGMAX', color='b')
    sorting_plot(CNF_empirical_dict, label='CNF', color='c')

    x = list(ground_truth_dict.keys())
    x_sort_arg = np.argsort(x)
    y = list(ground_truth_dict.values())
    x = [x[i] for i in x_sort_arg]
    y = [y[i] for i in x_sort_arg]
    plt.plot(x, y, color='r', label='stair')

    # plt.title('100samples.pk')
    plt.legend()
    plt.show()
    plt.close()

    # U posU ratio and S are parameters that will define the stair function
    # stair_histo = make_stair_prob(U, posU, ratio, S)
    # print(np.sum(list(stair_histo.values())))
    # print(stair_histo)
