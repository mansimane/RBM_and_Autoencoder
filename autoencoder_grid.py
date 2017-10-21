'''
Main file for autoencoder
'''
from load_mnist import *
from functions_ae import *
from config_ae import *
import matplotlib.pyplot as plt
import random

def plot_ce_train_valid (train_ce, valid_ce, hyper_para):
    date = time.strftime("%Y-%m-%d_%H_%M")
    plt.plot(train_ce)
    plt.plot(valid_ce)
    plt.title('AutoEncoder Error vs Epochs')
    hh = 'hl_size_' + str(hyper_para['hidden_layer_1_size'])
    a = '\tB_size=' + str(hyper_para['batch_size'])
    b =  '\tlr=' + str(hyper_para['learning_rate'])
    d = '\te: ' + str(hyper_para['epochs'])
    e = '\tDrop_out: ' + str(int(100*hyper_para['drop_out']))
    plt.suptitle(hh + a +b  + d + e)
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Error')
    plt.legend(['Train Cross Entropy', 'Valid Cross Entropy'], loc='upper right')
    if hyper_para['drop_out'] == 0.0:
        plt.savefig('./figures/5g/ae/Q5_g_error' + hh + '_' + date + '.png')
    else:
        plt.savefig('./figures/5g/dae/Q5_g_error'+ hh + '_' + date + '.png')
    plt.close()

def visualize (w1, hyper_para, show_flag, epoch) :
    date = time.strftime("%Y-%m-%d_%H_%M_%s")

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
        hh = 'hl_size_' + str(hyper_para['hidden_layer_1_size'])
        a = '\tB_size=' + str(hyper_para['batch_size'])
        b = '\tlr=' + str(hyper_para['learning_rate'])
        d = '\tepochs: ' + str(hyper_para['epochs'])
        if show_flag == 0:
            d = '\tepochs: ' + str(epoch)
        e = '\tDrop_out: ' + str(int(100 * hyper_para['drop_out']))
        plt.suptitle(hh + a + b  + d + e)
        if hyper_para['drop_out'] == 0.0:
            plt.savefig('./figures/5g/ae/Q5_g_weights' + hh + '_'+ date + '.png')
        else:
            plt.savefig('./figures/5g/dae/Q5_g_weights' + hh + '_' + date + '.png')
        plt.close()

# Load Training Data
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist()
no_of_train_samples = len(ytrain)


# Random seed for the run
random_seed = hyper_para['random_seed']
mu = hyper_para['w_init_mu']
sigma = hyper_para['w_init_sig']

J_train = 0
J_valid = 0
train_ce = []   #Cross Entropy = CE
valid_ce = []

indices = range(no_of_train_samples)
random.shuffle(indices)
batch_size = hyper_para['batch_size']
epochs = hyper_para['epochs']
max_iter = no_of_train_samples / batch_size       #Iterations within epoch for training

hlayer_size_grid = np.array([50,100,200,500])
for hh in range(0,hlayer_size_grid.shape[0]):

    hyper_para['hidden_layer_1_size'] = hlayer_size_grid[hh]
    print 'Starting HL size  ' + str(hyper_para['hidden_layer_1_size'])
    param = initialize_weights(hyper_para)

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

    plot_ce_train_valid(train_ce, valid_ce, hyper_para)
    visualize(param['w1'], hyper_para, 1, 2)
