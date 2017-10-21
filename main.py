import numpy as np
from PIL import Image
from load_mnist import *
from functions_final import *
from config import *
import matplotlib.pyplot as plt
import random
from plots_final import *
from visual import *

import time

# Load Training Data
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist()
no_of_train_samples = len(ytrain)


# Random seed for the run
random_seed = hyper_para['random_seed']
mu = hyper_para['w_init_mu']
sigma = hyper_para['w_init_sig']

np.random.seed(random_seed)

w = np.random.normal(mu, sigma, (input_layer_size * hidden_layer_1_size))
w = w.reshape(input_layer_size, hidden_layer_1_size)

b = np.random.normal(mu, sigma, (1, hidden_layer_1_size))

c = np.random.normal(mu, sigma, (1, input_layer_size))

param = {'w': w, 'b': b, 'c': c}

no_of_train_samples = len(ytrain)
h = np.zeros((1, hyper_para['hidden_layer_1_size']))
h.astype(float)

# Variables Storing Results
J_train = 0.0
J_valid = 0.0
train_ce = []   #Cross Entropy = CE
valid_ce = []

# learning iterations
indices = range(no_of_train_samples)
random.shuffle(indices)
batch_size = hyper_para['batch_size']
epochs = hyper_para['epochs']
max_iter = no_of_train_samples / batch_size       #Iterations within epoch for training

for epoch in range(epochs):
    for step in range(max_iter):
        # get mini-batch and setup the cnn with the mini-batch
        start_idx = step * batch_size % no_of_train_samples
        end_idx = (step + 1) * batch_size % no_of_train_samples

        if start_idx > end_idx:
            random.shuffle(indices)
            continue
        idx = indices[start_idx: end_idx]

        x_p, h_p = gibbs_step(param,  xtrain[idx, :], hyper_para)
        param = update_param(param, x_p, xtrain[idx, :], hyper_para)

    J_train = loss_calc(param, xtrain, ytrain, hyper_para)
    J_valid = loss_calc(param, xvalidate, yvalidate, hyper_para)

    print 'epoch', epoch, '\tTrain CE', J_train, '\t\tValid CE', J_valid

    train_ce.append(J_train)
    valid_ce.append(J_valid)

    if (epoch > 49) & (epoch % 50 ==0):
        save_obj(param, 'param', str(epoch))
        visualize(param['w'], hyper_para, epoch, 0)

    if ( epoch > 40):
        if ( abs(J_valid -J_train) < 2.2):
            save_obj(param, 'param', str(epoch))
            visualize(param['w'], hyper_para, epoch, 0)

plot_ce_train_valid(train_ce, valid_ce, hyper_para)
visualize(param['w'],hyper_para, epoch, 1)

