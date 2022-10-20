
import os
from plot_utils import plot_stat, put_on_plot
from statistic.generate_statistics import get_ranking_results

# initial plotting


def generating_S_rank_plots(list_of_title_q, list_of_results_stats, Bs, U):

    print('Generating S plots...')
    plotting_dict_algo = {}
    for i, title in enumerate(list_of_title_q):
        plotting_dict_algo[title] = list_of_results_stats[0][i]
    put_on_plot(Bs, plotting_dict_algo)
    prefix_title = 'U_' + str(U) + '_m_' + str(m)
    prefix_title = os.path.join('figures', prefix_title)
    plot_stat(prefix_title+'_algo_S.pdf', 'Bins',
              'Empirical Total Variation Error')

    plotting_dict_random = {}
    for i, title in enumerate(list_of_title_q):
        plotting_dict_random[title] = list_of_results_stats[1][i]
    put_on_plot(Bs, plotting_dict_random)

    # error of _____ w.r.t ground truth
    # for no temper, samples are generated from uniform dist
    # for heavily temper, samples are generated from heavily tempered distribution
    # thus, the gen model should have an easier time distinguishing the heavily
    # tempered case and will easily give it a lower rank

    plot_stat(prefix_title+'_random_S.pdf', 'Bins',
              'Empirical total variation error')

    print('Generating ranking plots...')

    algo_ranking_results_all_trials_all_Bs = []
    for i in range(len(Bs)):
        list_at_B = [q[i] for q in list_of_results_stats[0]]
        algo_ranking_results_all_trials_all_Bs.append(
            get_ranking_results(list_at_B))

    random_ranking_results_all_trials_all_Bs = []
    for i in range(len(Bs)):
        list_at_B = [q[i] for q in list_of_results_stats[1]]
        random_ranking_results_all_trials_all_Bs.append(
            get_ranking_results(list_at_B))

    put_on_plot(Bs, {'algo': algo_ranking_results_all_trials_all_Bs,
                'random': random_ranking_results_all_trials_all_Bs})
    plot_stat(prefix_title + '_ranking.pdf',
              'Bins', 'Kendall tau distance')
