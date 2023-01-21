# -*- coding: utf-8 -*-
"""AlphaPrecision_BetaRecall_EvalMetric.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cT2I5-vTYdqk7OQrsteeA1ikfPl2DQaf

# Based on the Alaa 2022 Code
"""

"""# Function Imports"""

from metrics.prdc import compute_prdc
import numpy as np
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
import pickle as pkl
import numpy as np
import io
import pandas as pd


from representations.OneClass import * 
from metrics.evaluation import *

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


def compute_metrics(X,Y, nearest_k = 5, model = None):
    # print(type(X))
    # print(type(Y))
    results = compute_prdc(X,Y, nearest_k)
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
    return X[:20000,:]

def get_stair():
  # placeholder, returns dummy array
  # shape: (100000, 6) <class 'numpy.ndarray'>
  test = np.ones((100000, 6), int)

  return test

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

def compare(X,Y):
    print("* Comparing with test arrays, to ensure methods work locally")
    print("* Will not return favorable results as origional distribution (the stair array) not correctly chosen in comparison")
    model = None
    res_, model = compute_metrics(X,Y, model=model)
    print(res_)
    return res_
print("----------------------------------------------------------")
print("Terminology Used in Results:")
print("Precision - classical def, supposedly same as Sajjadi 2018")
print("Recall - classical def, supposedly same as Sajjadi 2018")
print("Density - rebranded alpha precision in Alaa paper")
print("Coverage - rebranded beta recall in Alaa paper")
print("DPA/DCB - not used in the paper")
print("Mean Auth - authentication score, used in Alaa paper")
print("----------------------------------------------------------")
# unsure what to make the "stair function here" so just filled with ones
small_one = make_arr_small(get_100_pickle())
# print(small_one.shape)
small_stair = make_arr_small(get_stair())
# print(small_stair.shape)
result = compare(small_one,  small_stair)
# print(result)
print("----------------------------------------------------------")
given_ex_result = call_compute_metrics(1000, 64)