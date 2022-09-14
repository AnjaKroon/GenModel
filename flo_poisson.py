from discrete import makeUniProbArr, errFunct, genValArr, sampleSpecificProbDist
from gen_S import empirical_dist, genSstat
from plot_utils import plot_S_stat
from sampling.poisson import poisson_empirical_dist
import sys
import numpy as np
import random 

if __name__ == '__main__':
    # Set the random seed
    np.random.seed(32)
    random.seed(321)

    testCase = 2 # should be 1 or 2 depending on whether you want to run the program standalone or with a .sh script
    if testCase ==1:
        if len(sys.argv) != 6 :
            print("Usage:", sys.argv[0], "U m e b t")
            sys.exit() 
        #U = int(sys.argv[1])
        m = int(sys.argv[2])
        e = float(sys.argv[3])/100 # recall this value has been multiplied by 100 in sh script
        b = int(sys.argv[4])
        t = int(sys.argv[5])
        trials = t
    
    if testCase == 2:
        #U = 100
        m = 1000
        e = 0.1  # recall this value has been multiplied by 100 in sh script
        b = 50
        trials = 50

    S_uni = []
    S_uni_poisson = []
    S_tempered = []
    S_tempered_poisson = []
    list_U = [int((i+3)*10) for i in range(10)] # 30, 40, 50, 60, 70, 80, 90, 100, 110, 120
    for U in list_U:
        uni_prob_arr = makeUniProbArr(U)
        # uniform
        S_uni_trials = []
        S_uni_poisson_trials = []
        for _ in range(trials): 
            new_samples = sampleSpecificProbDist(genValArr(U), uni_prob_arr, m)
            p_emp_dependent = empirical_dist(
                U, m, sampleSpecificProbDist(genValArr(U), uni_prob_arr, m))
            s_statistic = genSstat(p_emp_dependent, U)
            S_uni_trials.append(s_statistic)
            p_emp_dependent = poisson_empirical_dist(
                U, m, new_samples, lambda m: sampleSpecificProbDist(genValArr(U), uni_prob_arr, m))
            s_statistic = genSstat(p_emp_dependent, U)
            S_uni_poisson_trials.append(s_statistic)

        S_uni.append(S_uni_trials)
        S_uni_poisson.append(S_uni_poisson_trials)
        # tempered

        S_tempered_trials = []
        S_tempered_poisson_trials = []
        updated_prob_arr = errFunct(U, uni_prob_arr, e, b)
        for _ in range(trials):
            new_samples = sampleSpecificProbDist(
                genValArr(U), updated_prob_arr, m)
            p_emp_dependent = empirical_dist(U, m, new_samples)
            s_statistic = genSstat(p_emp_dependent, U)
            S_tempered_trials.append(s_statistic)
            p_emp_dependent = poisson_empirical_dist(
                U, m, new_samples, lambda m: sampleSpecificProbDist(genValArr(U), updated_prob_arr, m))
            s_statistic = genSstat(p_emp_dependent, U)
            S_tempered_poisson_trials.append(s_statistic)
        S_tempered.append(S_tempered_trials)
        S_tempered_poisson.append(S_tempered_poisson_trials)


    
    lines = {'g.t.': S_uni,
             'g.t. with poisson.': S_uni_poisson,
             'e=0.1': S_tempered,
             'e=0.1 with poisson.': S_tempered_poisson}

    
    plot_S_stat(x=list_U, dict_y=lines, title='m_' +
                str(m) + '_e_'+str(e)+'_trials_'+str(trials)+'_S.pdf')
    # U m e b "S of uniform": NUMBER "S of uniform with poisson":NUMBER "S of tempered":NUMBER
