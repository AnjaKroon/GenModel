
from plot_utils import plot_stat, put_on_plot
import sys
import numpy as np
import random
from statistic.generate_statistics import get_S


if __name__ == '__main__':
    # Set the random seed
    np.random.seed(2)
    random.seed(3)

    testCase = 2  # should be 1 or 2 depending on whether you want to run the program standalone or with a .sh script
    if testCase == 1:
        if len(sys.argv) != 5:  # changed from 6 to make it work with the U
            print("Usage:", sys.argv[0], "U m e b t")
            sys.exit()
        #U = int(sys.argv[1])
        m = int(sys.argv[1])
        # recall this value has been multiplied by 100 in sh script
        e = float(sys.argv[2])/100
        b = int(sys.argv[3])
        t = int(sys.argv[4])
        trials = t
        # add bins var

    if testCase == 2:
        # list_U = [10000]
        # m = 2000
        e = 0.1
        b = 100
        trials = 50
        bins = 3

    S_uni = []
    S_uni_poisson = []
    S_tempered = []
    S_tempered_poisson = []

    S_uni_binned = []
    S_uni_poisson_binned = []
    S_tempered_binned = []
    S_tempered_poisson_binned = []

    rank = []
    rank_poisson = []
    rank_binned = []
    rank_poisson_binned = []
    list_U = [100, 1000]  # 1e10
    list_M = [100]
    # for m in list_M:
    for m in list_M:
        print("for this round m is ", m)
        for U in list_U:

            # BINNED CASE
            S_binned_uni_U = get_S(trials, U, m, tempered=False,
                                   e=e, b=b, B=int(U/bins), with_poisson=False)
            S_binned_uni_poisson_U = get_S(
                trials, U, m, tempered=False,  e=e, b=b, B=int(U/bins), with_poisson=True)
            # tempered
            S_binned_tempered_U = get_S(
                trials, U, m, tempered=True,  e=e, b=b, B=int(U/bins), with_poisson=False)
            S_binned_tempered_poisson_U = get_S(
                trials, U, m, tempered=True,  e=e, b=b, B=int(U/bins), with_poisson=True)

            # rank_uniform_bin vs rank_tempered_bin
            # uniform
            # S_uni_U = get_S(trials, U, m, tempered=False, with_poisson=False)
            # S_uni_poisson_U = get_S(
            #     trials, U, m, tempered=False, with_poisson=True)
            # tempered
            # S_tempered_U = get_S(trials, U, m, tempered=True, with_poisson=False)
            # S_tempered_poisson_U = get_S(
            #     trials, U, m, tempered=True, with_poisson=True)

            # count the number of time s of ground truth is smaller than tempered version
            # fraction_rank = np.mean([S_uni_U[i] < S_tempered_U[i]
            #                         for i in range(trials)])
            # fraction_poisson_rank = np.mean(
            #     [S_uni_poisson_U[i] < S_tempered_poisson_U[i] for i in range(trials)])

            # BINNED
            fraction_binned_rank = np.mean([S_binned_uni_U[i] < S_binned_tempered_U[i]
                                            for i in range(trials)])
            fraction_poisson_binned_rank = np.mean(
                [S_binned_uni_poisson_U[i] < S_binned_tempered_poisson_U[i] for i in range(trials)])

            '''
            if (fraction_poisson_rank <= 0.5):  # how many times is the uniform better than the tempered
                # 50 is saying we are not sure which one is better
                print(fraction_poisson_rank)
                print("For U:"+str(U)+" m:"+str(m)+" e:"+str(e)+" b:"+str(b)+" t:" +
                    str(trials) + " fraction_poisson_rank did not meet the threshold of 0.5.")
            else:
                print("For U:"+str(U)+" m:"+str(m)+" e:"+str(e)+" b:"+str(b)+" t:" +
                    str(trials)+"fraction_poisson_rank: " + str(fraction_poisson_rank))
            '''

            # uniform
            # S_uni.append(S_uni_U)
            # S_uni_poisson.append(S_uni_poisson_U)

            # tempered
            # S_tempered.append(S_tempered_U)
            # S_tempered_poisson.append(S_tempered_poisson_U)

            # uniform BINNED
            S_uni_binned.append(S_binned_uni_U)
            S_uni_poisson_binned.append(S_binned_uni_poisson_U)

            # tempered BINNED
            S_tempered_binned.append(S_binned_tempered_U)
            S_tempered_poisson_binned.append(S_binned_tempered_poisson_U)

            # rank.append(fraction_rank)
            # rank_poisson.append(fraction_poisson_rank)
            rank_binned.append(fraction_binned_rank)
            rank_poisson_binned.append(fraction_poisson_binned_rank)

            # lines_S = {'g.t.': S_uni,
            # 'e=0.1': S_tempered}
            # lines_S_poisson = {
            # 'g.t. with poisson.': S_uni_poisson,
            # 'e=0.1 with poisson.': S_tempered_poisson}
            # lines_rank = {'GT v. Tem m = '+str(m): rank,'GT v. Tem Poi. m = ' +str(m): rank_poisson}
            # lines_rank = {'GT v. Tem m = '+str(m): rank}
            # lines_S_binned = {'g.t.': S_uni_binned,
            #                 'e=0.1': S_tempered_binned}
            # lines_S_poisson_binned = {
            #     'g.t. with poisson.': S_uni_poisson_binned,
            # 'e=0.1 with poisson.': S_tempered_poisson_binned}
            lines_rank_binned = {
                'GT v. Tem m = '+str(m): rank_binned,
                'GT v. Tem Poi. m = ' + str(m): rank_poisson_binned}
        put_on_plot(x=list_U, dict_y=lines_rank_binned)
        rank_binned = []  # clearing the array
        rank_poisson_binned = []  # clearing the array
        '''
        plot_S_stat(x=list_U, dict_y=lines_S_poisson, title='m_' +
                    str(m) + '_e_'+str(e)+'_trials_'+str(trials)+'_Spoisson.pdf', xlabel='|U|', ylabel='S')
        plot_S_stat(x=list_U, dict_y=lines_S, title='m_' +
                    str(m) + '_e_'+str(e)+'_trials_'+str(trials)+'_S.pdf', xlabel='|U|', ylabel='S')
        
        # Ranking
        plot_S_stat(x=list_U, dict_y=lines_rank, title='m_' +
                    str(m) + '_e_'+str(e)+'_trials_'+str(trials)+'_ranking.pdf', xlabel='|U|', ylabel='\% of accurate ranking')

        plot_S_stat(x=list_U, dict_y=lines_S_poisson_binned, title='binned_m_' +
                    str(m) + '_e_'+str(e)+'_trials_'+str(trials)+'_Spoisson.pdf', xlabel='|U|', ylabel='S')
        plot_S_stat(x=list_U, dict_y=lines_S_binned, title='binned_m_' +
                    str(m) + '_e_'+str(e)+'_trials_'+str(trials)+'_S.pdf', xlabel='|U|', ylabel='S')
'''
    # Ranking BINNED
    plot_stat(title='BINS_'+str(bins)+'_Multiple_U_Multiple_M_e_'+str(e)+'_b_' + str(b) +
              '_t_'+str(trials)+'_ranking.pdf', xlabel='|U|', ylabel='\% of accurate ranking')

    # plot_S_stat(x=list_U, dict_y=lines_rank, title='UNBINNED_Multiple_U_Multiple_M_e_'+str(e)+'_b_'+ str(b)+'_t_'+str(trials)+'_ranking.pdf', xlabel='|U|', ylabel='\% of accurate ranking')
    # I think this will still overwrite as plot is being called. I think it does need to happen in plot_utils.py
    # U m e b "S of uniform": NUMBER "S of uniform with poisson":NUMBER "S of tempered":NUMBER
