import numpy as np
from PIL import Image
from load_mnist import *
from functions_final import *
from config import *
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time

def visualize (w1, hyper_para, epoch, show_flag ) :
    date = time.strftime("%Y-%m-%d_%H_%M")

    if w1.shape[0] == 784:
        if w1.shape[1] == 50:
            nrow = 10
            ncol = 5
        if w1.shape[1] == 100:
            nrow = 10
            ncol = 10
        if w1.shape[1] == 200:
            nrow = 20
            ncol = 10
        if w1.shape[1] == 500:
            nrow = 20
            ncol = 10

        fig = plt.figure(figsize=(ncol + 1, nrow + 1))

        gs = gridspec.GridSpec(nrow, ncol,
                               wspace=0.0, hspace=0.0,
                               top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                               left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
        cnt = 0
        for i in range(nrow):
            for j in range(ncol):
                im = np.reshape(w1[:, cnt], (28, 28))
                ax = plt.subplot(gs[i, j])
                ax.imshow(im)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                cnt = cnt + 1
        #plt.title('Weights at layer 1')
        kk = '_k_' + str(hyper_para['k'])
        #hh = 'hl_size_' + str(hyper_para['hidden_layer_1_size'])
        plt.suptitle( kk + '\tBatch_size=' + str(hyper_para['batch_size']) + '\tlearning_rate=' + str(
            hyper_para['learning_rate']) + '\tk=' + str(hyper_para['k']) + 'epoch_' + str(epoch))
        # if show_flag == 1:
        #     plt.show()
    # plt.savefig('./figures/5b/Q5_b_weights' + hh+'_'+ date + '.png')
    plt.savefig('./figures/5b/Q5_b_weights' + kk+'_'+ date + '.png')
    plt.close()


def plot_ce_train_valid(train_ce, valid_ce, hyper_para):
    date = time.strftime("%Y-%m-%d_%H_%M")
    plt.plot(train_ce)
    plt.plot(valid_ce)
    plt.title('Error vs Epochs')
    #hh = 'hl_size_' + str(hyper_para['hidden_layer_1_size'])
    kk = '_k_' + str(hyper_para['k'])
    a = '\tBatch_size=' + str(hyper_para['batch_size'])
    b = '\tlearning_rate=' + str(hyper_para['learning_rate'])
    c = '\tk=' + str(hyper_para['k'])
    plt.suptitle(kk + a + b + c)
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Error')
    plt.legend(['Train Cross Entropy', 'Valid Cross Entropy'], loc='upper right')
    plt.savefig('./figures/5b/Q5_b_rbm_' + kk + '_'+ date + '.png')
    #plt.show()
    plt.close()


# Load Training Data
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist()
no_of_train_samples = len(ytrain)


# Random seed for the run
random_seed = hyper_para['random_seed']
mu = hyper_para['w_init_mu']
sigma = hyper_para['w_init_sig']

np.random.seed(random_seed)


no_of_train_samples = len(ytrain)
h = np.zeros((1, hyper_para['hidden_layer_1_size']))
h.astype(float)

# Variables Storing Results
J_train = 0.0
J_valid = 0.0

# learning iterations
indices = range(no_of_train_samples)
random.shuffle(indices)
batch_size = hyper_para['batch_size']
epochs = hyper_para['epochs']
max_iter = no_of_train_samples / batch_size       #Iterations within epoch for training

#hlayer_size_grid = np.array([50,100,200,500])
hlayer_size_grid = np.array([100])
k_grid = np.array([1,5,10,20])

for hh in range(0,hlayer_size_grid.shape[0]):
    for k in range(0, k_grid.shape[0]):

        hyper_para['hidden_layer_1_size'] = hlayer_size_grid[hh]
        hyper_para['k'] = k_grid[k]

        print 'Starting K = ', str(hyper_para['k']) + ' HL size  ' + str(hyper_para['hidden_layer_1_size'])
        train_ce = []  # Cross Entropy = CE
        valid_ce = []

        hidden_layer_1_size = hlayer_size_grid[hh]  # 100 hidden units
        w = np.random.normal(mu, sigma, (input_layer_size * hidden_layer_1_size))
        w = w.reshape(input_layer_size, hidden_layer_1_size)

        b = np.random.normal(mu, sigma, (1, hidden_layer_1_size))
        c = np.random.normal(mu, sigma, (1, input_layer_size))
        param = {'w': w, 'b': b, 'c': c}

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

            param_string = 'param_rbm_k_' + str(hyper_para['k'])
            if (epoch > 49) & (epoch % 50 == 0):
                save_obj(param, param_string, str(epoch))
                visualize(param['w'], hyper_para, epoch, 0)

            train_ce.append(J_train)
            valid_ce.append(J_valid)

        plot_ce_train_valid(train_ce, valid_ce, hyper_para)
        visualize(param['w'], hyper_para, epoch, 1)

