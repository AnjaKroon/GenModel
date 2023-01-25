# -*- coding: utf-8 -*-
"""AlphaPrecision_BetaRecall_EvalMetric.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cT2I5-vTYdqk7OQrsteeA1ikfPl2DQaf

# Based on the Alaa 2022 Code
"""

"""# Function Imports"""

import math
from metrics.prdc import compute_prdc
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle as pkl
import numpy as np
import io
import pandas as pd


from representations.OneClass import * 
from metrics.evaluation import *
from random import shuffle

nearest_k = 5
params  = dict({"rep_dim": None, 
                "num_layers": 2, 
                "num_hidden": 200, 
                "activation": "ReLU",
                "dropout_prob": 0.5, 
                "dropout_active": False,
                "train_prop" : 1,
                "epochs" : 100,
                "warm_up_epochs" : 10,
                "lr" : 1e-3,
                "weight_decay" : 1e-2,
                "LossFn": "SoftBoundary"})   

hyperparams = dict({"Radius": 1, "nu": 1e-2})



def plot_all(x, res, x_axis):
    print(x_axis)
    if type(res) == type([]):
        plot_legend = False
        res = {'0':res}
    else:
        plot_legend = True
    exp_keys = list(res.keys())
    print(res)
    metric_keys = res[exp_keys[0]][0].keys() 
    for m_key in metric_keys:
        for e_key in exp_keys:
          y = [res[e_key][i][m_key] for i in range(len(x))]
          plt.plot(x, y, label=e_key)
        
        plt.ylabel(m_key)
        plt.ylim(bottom=0)
        plt.xlabel(x_axis) 
        if plot_legend:
            plt.legend()
        plt.show()


def compute_metrics(X,Y, nearest_k = 5, model = None, distance=None):
    
    def get_category_bias():
        all_cats = []
        for row in range(Y.shape[0]):       # for each row in each array
            cat = 0                         # set category bias to 0
            permutation = False
            summation = 0
            # need to add check to see if permutation
            # print(Y[row, :])
            empty_list = []
            for each in Y[row, :]:
                if each not in empty_list: empty_list.append(each)
            if len(empty_list) == 6: permutation = True
            # if permutation == True: print('permu True')    
            if Y[row, 0] > Y[row, -1] and permutation == True:
                cat = 1
            elif Y[row, 0] < Y[row, -1] and permutation == True:
                cat = 2
            if permutation == False: cat = 3
            print(cat)
            all_cats.append(cat)
        return all_cats
    
    all_biases = get_category_bias()
    print(len(all_biases))
        

    results = compute_prdc(X,Y, all_biases, nearest_k, distance)
    if model is None:
        #these are fairly arbitrarily chosen
        params["input_dim"] = X.shape[1]
        params["rep_dim"] = X.shape[1]        
        hyperparams["center"] = torch.ones(X.shape[1])
        # print(type(model))
        model = OneClassLayer(params=params, hyperparams=hyperparams)
        # print(type(model))
        model.fit(X,verbosity=False)

    X_out = model(torch.tensor(X).float()).float().detach().numpy()
    Y_out = model(torch.tensor(Y).float()).float().detach().numpy()

    # sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor)
    # attempted non true version
    # trying to put it to a torch
    # error on float not being a double so changed float to a double -- not possible
    # maybe an issue with types? trying to print out the types now
    # Current error is occuring with the model function. Let's see where that's coming from

    #X_ten = torch.tensor(X)
    #Y_ten = torch.tensor(Y)

    #print(type(X_ten))
    #print(type(Y_ten))

    #print(type( X_ten.clone().detach() ))

    #X_test = model(X_ten.clone().detach())

    #X_out = model(X_ten.clone().detach()).float().detach().numpy()
    #Y_out = model(Y_ten.clone().detach()).float().detach().numpy()
    
    # print(type(X_out))
    # print(type(Y_out))
    # print(type(model.c))

    #alphas, alpha_precision_curve, beta_coverage_curve, Delta_precision_alpha, Delta_coverage_beta, (thresholds, authen) = compute_alpha_precision(X_out, Y_out, model.c)
    alphas, alpha_precision_curve, beta_coverage_curve, Delta_precision_alpha, Delta_coverage_beta, a = compute_alpha_precision(X_out, Y_out, model.c)
    # print(a)
    results['Dpa'] = Delta_precision_alpha
    results['Dcb'] = Delta_coverage_beta
    # results['mean_aut'] = np.mean(authen)
    results['mean_aut'] = np.mean(a)        # I don't know if this would change the program intent
    return results, model

"""# Define Orig Data and Gen Data

Take pickle data from gen model and use metrics to compare generated data to stair distribution.
"""

def comparing_all_gen_models():
    small_random = make_arr_small(get_random_input())
    small_train_input = make_arr_small(get_train())

    argmaxAR_input = open('data/100sample_Data/100sample_argmaxAR.pk','rb')
    argmaxAR = make_arr_small(pkl.load(argmaxAR_input))

    CDM_input = open('data/100sample_Data/100sample_CDM.pk','rb')
    CDM = make_arr_small(pkl.load(CDM_input))

    CNF_input = open('data/100sample_Data/100sample_CNF.pk','rb')
    CNF = make_arr_small(pkl.load(CNF_input))
    
    FCDM_input = open('data/100sample_Data/100sample_FCDM.pk','rb')
    FCDM = make_arr_small(pkl.load(FCDM_input))

    print("Experiment 1: Ground Truth vs Random Output")
    print(small_train_input.shape, small_random.shape)
    print(compare(small_train_input, small_random, distance='hamilton'))

    print("Experiment 2: Ground Truth vs argmaxAR Output")
    print(compare(small_train_input, argmaxAR, distance='hamilton'))

    print("Experiment 3: Ground Truth vs CDM Output")
    print(compare(small_train_input, CDM, distance='hamilton'))

    #print("Experiment 4: Random Input vs CDM Output")
    #print(compare(small_random_input, CDM, distance='hamilton'))

    print("Experiment 4: Ground Truth vs CNF Output")
    print(compare(small_train_input, CNF, distance='hamilton'))

    #print("Experiment 6: Random Input vs CNF Output")
    #print(compare(small_random_input, CNF, distance='hamilton'))

    print("Experiment 5: Ground Truth vs FCDM Output")
    print(compare(small_train_input, FCDM, distance='hamilton'))

    #print("Experiment 8: Random Input vs FCDM Output")
    #print(compare(small_random_input, FCDM, distance='hamilton'))
    return 

def get_100_pickle():
  # pulls 100sample.pk file in and puts into np array
  # shape: (100000, 6) <class 'numpy.ndarray'>
  infile = open('100sample.pk','rb')
  hund_pickle_samples = pkl.load(infile)
  return hund_pickle_samples

def get_1_pickle():
    # pulls 100sample.pk file in and puts into np array
    # shape: (100000, 6) <class 'numpy.ndarray'>
    infile = open('1sample.pk','rb')
    one_pickle_sample = pkl.load(infile)

    return one_pickle_sample

def make_arr_small(X):
    return X[:10000,:]

def make_arr_medium(X):
    return X[:20000,:]

def get_train():
    infile = open('train.pk','rb')
    one_pickle_sample = pkl.load(infile)
    return one_pickle_sample

def get_random_input():
    rand_input = np.array
    output = np.array([[0,1,2,3,4,5]])
    for i in range(20000):
        rand_input = np.random.permutation([0,1,2,3,4,5])
        # print(rand_input)
        output = np.append(output, [rand_input], axis=0)
    output = output[1:]
    # print(output.shape)
    return output

"""# Calling Compute Metrics"""

# This is a method from the "toy metric evaluation code"
# This code is included here as example code
def translation_test(d=64, n=1000, step_size=0.1):
    X = np.random.randn(n,d) # returns samples from normal dist, 
    Y_0 = np.random.randn(n,d)
    print(X.shape)
    print(Y_0.shape)

    X_outlier = X.copy()
    X_outlier[-1] = np.ones(d)

    res = []
    res_outr = []
    res_outf = []

    # translation
    mus = np.arange(0,0.2+step_size,step_size)
    model = None
    for i, mu in enumerate(mus):
        Y = Y_0 + mu
        res_, model = compute_metrics(X,Y, model=model)
        res.append(res_)

    plot_all(mus, res, r'$\mu$')

def call_compute_metrics(n,d):
    print("* Now demonstrating example with distributions which are similar.")
    print("* Will produce favorable results as the distributions are close enough")
    X = np.random.randn(n,d) # returns samples from normal dist, 
    Y = np.random.randn(n,d)
    
    model = None
    res_, model = compute_metrics(X,Y, model=model)
    print(res_)
    return res_

def compare(X,Y, distance=None):
    model = None
    res_, model = compute_metrics(X,Y, model=model, distance=distance)
    # print(res_)
    return res_
print("----------------------------------------------------------")
print("Terminology Used in Results:")
print("Precision - classical def, supposedly same as Sajjadi 2018")
print("Recall - classical def, supposedly same as Sajjadi 2018")
print("Density - rebranded alpha precision in Alaa paper")
print("Coverage - rebranded beta recall in Alaa paper")
print("DPA/DCB - not used in the paper")
print("Mean Auth - authentication score, used in Alaa paper")

# print("----------------------------------------------------------")
# small_train = make_arr_small(get_train())
#print("small train input", small_train.shape)
# print(small_train)

#small_output = make_arr_small(get_100_pickle())
#print("small output", small_output.shape)
# print(small_output)
#m = small_output.shape[0]

#train_vs_output = compare(small_train, small_output, distance='hamilton')
#print("For the Training Data Comparison")
#print(train_vs_output)

# # print("----------------------------------------------------------")
# random_input = get_random_input()
# small_output = make_arr_medium(get_100_pickle())
#print("random train input", random_input.shape)
#print("small output", small_output.shape)
# print("medium output", small_output.shape)
# print(random_input)
# print("For the Random Comparison")
# random_vs_output = compare(random_input, small_output, distance='hamilton')
# print(random_vs_output)

# print("----------------------------------------------------------")
# given_ex_result = call_compute_metrics(1000, 64)

print("----------------------------------------------------------")
comparing_all_gen_models()


