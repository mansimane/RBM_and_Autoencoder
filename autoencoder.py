'''
Main file for autoencoder
'''
from load_mnist import *
from functions_ae import *
from config_ae import *
import matplotlib.pyplot as plt
import random


# Load Training Data
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist()
no_of_train_samples = len(ytrain)


# Random seed for the run
random_seed = hyper_para['random_seed']
mu = hyper_para['w_init_mu']
sigma = hyper_para['w_init_sig']

param = initialize_weights(hyper_para)
J_train = 0
J_valid = 0
train_ce = []   #Cross Entropy = CE
valid_ce = []

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
        param_grad = grad_calc(param, xtrain[idx, :], hyper_para)
        param = update_param(param, param_grad, hyper_para)

    J_train = loss_calc(param, xtrain, ytrain, hyper_para)
    J_valid = loss_calc(param, xvalidate, yvalidate, hyper_para)

    print 'epoch', epoch, '\tTrain CE', J_train, '\t\tValid CE', J_valid

    train_ce.append(J_train)
    valid_ce.append(J_valid)
    #save parameters
    if (epoch > 100) and (epoch % 50 ==0):
        if hyper_para['drop_out'] == 0:
            save_obj(param, 'param_ae', str(epoch))
            visualize(param['w1'], hyper_para,0, epoch )
        else:
            save_obj(param, 'param_dae', str(epoch))
            visualize(param['w1'], hyper_para, 0, epoch)
save_obj(param, 'param_dae', str(epoch))

plot_ce_train_valid(train_ce, valid_ce, hyper_para)
visualize(param['w1'], hyper_para,1, 2)
