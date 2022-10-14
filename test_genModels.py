import math
from tqdm import tqdm
from discrete import makeUniProbArr, prob_array_to_dict
from stair import make_stair_prob
from statistic.generate_statistics import perform_binning_and_compute_stats, genSstat, generate_samples_scalable
import matplotlib.pyplot as plt
from plot_utils import plot_stat, put_on_plot
import scipy
import numpy as np
import random
from read_pickle import read_pickle_file
from stair import samples_to_histo, build_ground_truth_dict, get_converting_dict, convert_key_sequence_to_int


def get_ranking_results(all_models_list_stats):
    number_of_models = len(all_models_list_stats)
    number_of_trials = len(all_models_list_stats[0])
    ground_truth_ranking = list(range(number_of_models))
    kendalltau_ranking_metric_all_trials = []

    for i in range(number_of_trials):
        trial_i_all_models = [trials[i] for trials in all_models_list_stats]
        ranking_i = np.argsort(trial_i_all_models)
        kendalltau = scipy.stats.kendalltau(ranking_i, ground_truth_ranking)[
            0]  # only takes the statistic, higher is better
        kendalltau_ranking_metric_all_trials.append(kendalltau)
    return kendalltau_ranking_metric_all_trials


if __name__ == '__main__':
    # Set the random seed
    np.random.seed(3)
    random.seed(3)

    init_e = 0.1
    init_b = 30
    trials = 50
    
    distribution_type = 'STAIRS'  # STAIRS
    Bs = [3,4,5,6]
    power_base = 6
    list_U = [power_base**power_base]
    list_M = [100000]
    # for m in list_M:
    for m in list_M:
        print("for this round m is ", m)
        for U in list_U:
            print("and U is ", U)

            stat_uni = []
            stat_temper = []
            stat_mid_temper = []
            stat_easy_temper = []
            stat_uni_baseline = []
            stat_temper_baseline = []
            stat_mid_temper_baseline = []
            stat_easy_temper_baseline = []
            all_U = []
            
           # plot_stat('PMF_Uniform_Stairs.pdf', 'U', 'Probability')
            # first, we generate all the samples here. The same samples should be reused for each B.
            # If it takes too much memory, we can put this in a for loop.
            
            
            
            # want to do the same as this but with the samples generated from the generative model
            # can use the perform binning & compute stats function
            # need to loop over the types of gen models
            # read data in with read_pickle
            # convert samples to a histo with samples_to_histo() 
            # perform_binning_and_compute_stats(samples_from_gen_model, ground_truth_p, U_for_samples, B=B, title)
                # Question: CAN I USE THE SAME GROUND_TRUTH_P THAT WAS USED BEFORE?
            # Compile all stats????
            # put_on_plot(Bs, {'GEN-MODEL-TYPE': stat_uni, 'samples from slightly tempered dist.': stat_temper,
                # 'samples from medium tempered dist': stat_mid_temper, 'samples from heavily tempered': stat_easy_temper})
            # compile_all_stats(genmodel_samples_list, stat_genmodelX, stat_groundtruth, U, B=B, title="gen model x")

            arg_max_samples_from_file = read_pickle_file('100sample.pk', './S_6_K_6/argmaxAR/13_08_2022__23_08/figure')
            arg_max_empirical_dict = samples_to_histo(arg_max_samples_from_file)
            # TODO: add the other elements later

            CDM_samples_from_file = read_pickle_file('100sample.pk', './S_6_K_6/CDM/07_08_2022__11_49/figure')
            CDM_empirical_dict = samples_to_histo(CDM_samples_from_file)

            CNF_samples_from_file = read_pickle_file('100sample.pk', './S_6_K_6/CNF/14_08_2022__10_54/figure')
            CNF_empirical_dict = samples_to_histo(CNF_samples_from_file)


            

            ground_truth_dict = build_ground_truth_dict() # same for all
            KEY_CONVERTING_DICT = get_converting_dict() # same for all

            arg_max_empirical_dict = convert_key_sequence_to_int(arg_max_empirical_dict, KEY_CONVERTING_DICT)
            CDM_empirical_dict = convert_key_sequence_to_int(CDM_empirical_dict, KEY_CONVERTING_DICT)
            CNF_empirical_dict =convert_key_sequence_to_int(CNF_empirical_dict, KEY_CONVERTING_DICT)
            ground_truth_dict = convert_key_sequence_to_int(ground_truth_dict, KEY_CONVERTING_DICT) # same for all

            # argmax_samples_list = 
            # CDM_samples_list = 
            # CNF_samples_list = 
            # FCDM_samples_list = 
            # Transformer_samples_list = 
            
            # replace with empirical from pickle
            ground_truth_samples_list = [ground_truth_dict]
            #ground_truth_samples_list = generate_samples_scalable(ground_truth_p,
                                                                  #trials, U, m, tempered=False, e=0, b=100)
            tempered_samples_list = [arg_max_empirical_dict]
            mid_tempered_samples_list = [CDM_empirical_dict]
            easy_tempered_samples_list = [CNF_empirical_dict]
            # tempered_samples_list = generate_samples_scalable(ground_truth_p,
                                                              #trials, U, m, tempered=True, e=init_e, b=init_b)
            # mid_tempered_samples_list = generate_samples_scalable(ground_truth_p,
                                                                  #trials, U, m, tempered=True, e=init_e*1.5, b=init_b)
            # easy_tempered_samples_list = generate_samples_scalable(ground_truth_p,
                                                                   #trials, U, m, tempered=True, e=init_e*2, b=init_b)

            
            for B in tqdm(Bs):  # For each bin granularity

                # this fonction takes the samples, compile statistics, then store the result in all_stat_list
                def compile_all_stats(all_samples_list, all_stat_list, baseline_all_stat_list,   U,  B, title):

                    S_trials_for_this_B_list = perform_binning_and_compute_stats(
                        all_samples_list, ground_truth_dict, U, B, stat_func=genSstat)
                    algo_stat = [i['B_algo'] for i in S_trials_for_this_B_list]
                    random_stat = [i['B_random']
                                   for i in S_trials_for_this_B_list]
                    all_stat_list.append(algo_stat)
                    baseline_all_stat_list.append(random_stat)
                    name = ("compile_stats_" + str(B) + str(title) + ".txt")
                    with open(name, "w") as output:
                        # writing baseline_all_stat_list into file -- should it be all_stat_list?
                        output.write(str(baseline_all_stat_list))

                # compile stats for ground truth
                # compile stats for ground truth
                # compile_all_stats(ground_truth_samples_list,
                #                   stat_uni, stat_uni_baseline, U, B=B, title="ground_truth_samples")
                # compile stats for tempered
                compile_all_stats(tempered_samples_list,
                                  stat_temper, stat_temper_baseline, U, B=B, title="tempered_samples")
                compile_all_stats(mid_tempered_samples_list, stat_mid_temper,
                                  stat_mid_temper_baseline, U, B=B, title="mid_tempered_samples")  # compile stats for tempered
                compile_all_stats(easy_tempered_samples_list, stat_easy_temper,
                                   stat_easy_temper_baseline, U, B=B, title="easy_tempered_samples")  # compile stats for tempered

            print('Generating S plots...')

            put_on_plot(Bs, { 'arg max': stat_temper,
                        'CDM': stat_mid_temper, 'CNF': stat_easy_temper})

            plot_stat('algo_S.pdf', 'Bins',
                      'Empirical Total Variation Error')

            put_on_plot(Bs, {'ground truth': stat_uni_baseline, 'arg max': stat_temper_baseline,
                         'CDM': stat_mid_temper_baseline, 'CNF': stat_easy_temper_baseline})
                         # error of _____ w.r.t ground truth
                         # for no temper, samples are generated from uniform dist
                         # for heavily temper, samples are generated from heavily tempered distribution
                         # thus, the gen model should have an easier time distinguishing the heavily 
                         # tempered case and will easily give it a lower rank

            plot_stat('random_S.pdf', 'Bins',
                      'Empirical total variation error')

            print('Generating ranking plots...')
            
            algo_ranking_results_all_trials_all_Bs = [get_ranking_results(
                [stat_uni[i], stat_temper[i], stat_mid_temper[i], stat_easy_temper[i]]) for i in range(len(Bs))]

            random_ranking_results_all_trials_all_Bs = [get_ranking_results(
                [stat_uni_baseline[i], stat_temper_baseline[i], stat_mid_temper_baseline[i], stat_easy_temper_baseline[i]]) for i in range(len(Bs))]
            #ranking_results_mean = [np.mean(ranking_results_all_trials) for ranking_results_all_trials in ranking_results_all_trials_all_Bs]

            put_on_plot(Bs, {'algo': algo_ranking_results_all_trials_all_Bs,
                        'random': random_ranking_results_all_trials_all_Bs})
            plot_stat('ranking.pdf', 'Bins','Kendall tau distance')
