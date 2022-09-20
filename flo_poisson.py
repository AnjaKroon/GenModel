import fractions
from discrete import makeUniProbArr, errFunct, genValArr, sampleSpecificProbDist
from gen_S import empirical_dist, genSstat
from plot_utils import plot_S_stat
from sampling.poisson import poisson_empirical_dist
import sys
import numpy as np
import random


def get_S(trials, U, m, tempered, with_poisson=True):
    uni_prob_arr = makeUniProbArr(U)
    prob_array = uni_prob_arr
    if tempered:
        prob_array = errFunct(U, uni_prob_arr, e, b)
    S_trials = []
    for _ in range(trials):
        new_samples = sampleSpecificProbDist(genValArr(U), prob_array, m)
        if with_poisson:
            p_emp = poisson_empirical_dist(
                U, m, new_samples, lambda m: sampleSpecificProbDist(genValArr(U), prob_array, m))
        else:
            p_emp = empirical_dist(
                U, m, sampleSpecificProbDist(genValArr(U), prob_array, m))
        s_statistic = genSstat(p_emp, U)
        S_trials.append(s_statistic)
    return S_trials


if __name__ == '__main__':
    # Set the random seed
    np.random.seed(2)
    random.seed(32)

    testCase = 2  # should be 1 or 2 depending on whether you want to run the program standalone or with a .sh script
    if testCase == 1:
        if len(sys.argv) != 6:
            print("Usage:", sys.argv[0], "U m e b t")
            sys.exit()
        #U = int(sys.argv[1])
        m = int(sys.argv[2])
        # recall this value has been multiplied by 100 in sh script
        e = float(sys.argv[3])/100
        b = int(sys.argv[4])
        t = int(sys.argv[5])
        trials = t

    if testCase == 2:
        list_U = [10000]
        m = 30000
        e = 0.1  
        b = 100
        trials = 50

    S_uni = []
    S_uni_poisson = []
    S_tempered = []
    S_tempered_poisson = []
    rank = []
    rank_poisson = []
    # list_U = [2000]
    for U in list_U:
        # copy two times
        # primarily looking at rank
        # rank_uniform_bin vs rank_tempered_bin
        # uniform
        S_uni_U = get_S(trials, U, m, tempered=False, with_poisson=False)
        S_uni_poisson_U = get_S(
            trials, U, m, tempered=False, with_poisson=True)
        # tempered
        S_tempered_U = get_S(trials, U, m, tempered=True, with_poisson=False)
        S_tempered_poisson_U = get_S(
            trials, U, m, tempered=True, with_poisson=True)

        # count the number of time s of ground truth is smaller than tempered version
        fraction_rank = np.mean([S_uni_U[i] < S_tempered_U[i]
                                for i in range(trials)])
        fraction_poisson_rank = np.mean(
            [S_uni_poisson_U[i] < S_tempered_poisson_U[i] for i in range(trials)])

        if (fraction_poisson_rank<=0.5): # how many times is the uniform better than the tempered
            # 50 is saying we are not sure which one is better
            print(fraction_poisson_rank)
            print("For U:"+str(U)+" m:"+str(m)+" e:"+str(e)+" b:"+str(b)+" t:"+str(trials)+ " fraction_poisson_rank did not meet the threshold of 0.5.")
        else:
            print("For U:"+str(U)+" m:"+str(m)+" e:"+str(e)+" b:"+str(b)+" t:"+str(trials)+"fraction_poisson_rank: "+ str(fraction_poisson_rank))
        # uniform
        S_uni.append(S_uni_U)
        S_uni_poisson.append(S_uni_poisson_U)
        # tempered
        S_tempered.append(S_tempered_U)
        S_tempered_poisson.append(S_tempered_poisson_U)
        rank.append(fraction_rank)
        rank_poisson.append(fraction_poisson_rank)
    lines_S = {'g.t.': S_uni,
               'g.t. with poisson.': S_uni_poisson,
               'e=0.1': S_tempered,
               'e=0.1 with poisson.': S_tempered_poisson}
    lines_rank = {
        'g.t. vs tempered': rank,
        'g.t. vs tempered with poisson': rank_poisson}
    plot_S_stat(x=list_U, dict_y=lines_S, title='m_' +
                str(m) + '_e_'+str(e)+'_trials_'+str(trials)+'_S.pdf', xlabel='|U|', ylabel='S')
   
    plot_S_stat(x=list_U, dict_y=lines_rank, title='m_' +
                str(m) + '_e_'+str(e)+'_trials_'+str(trials)+'_ranking.pdf', xlabel='|U|', ylabel='\% of accurate ranking')
    # U m e b "S of uniform": NUMBER "S of uniform with poisson":NUMBER "S of tempered":NUMBER
