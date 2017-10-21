'''
Contains functions related to autoencoder
'''
import copy
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
def initialize_weights(hyper_para):
    param = {}
    mu = hyper_para['w_init_mu']
    sigma = hyper_para['w_init_sig']

    input_layer_size = hyper_para['input_layer_size']
    output_layer_size = input_layer_size

    hidden_layer_size1 = hyper_para['hidden_layer_1_size']

    w1 = np.random.normal(mu, sigma, (input_layer_size * hidden_layer_size1))
    w1 = w1.reshape(input_layer_size, hidden_layer_size1)

    w2 = np.random.normal(mu, sigma, (hidden_layer_size1 * output_layer_size))
    w2 = w2.reshape(hidden_layer_size1, output_layer_size)

    b1 = np.random.normal(mu, sigma, (1, hidden_layer_size1))

    b2 = np.random.normal(mu, sigma, (1, output_layer_size))

    param['w1'] = w1
    param['w2'] = w2
    param['b1'] = b1
    param['b2'] = b2
    return param

def sigmoidGradient (z):
    g = np.multiply(z , (1-z))

    return g

def sigmoid_forward (z):
    g = 1.0/(1.0 + np.exp(-z))
    return g

def act_forward(input, w, b):
    """ Args:

        Returns:
    """
    a = input.dot(w) + b
    return a

def sigmoid_back (input, output, grad_prev ):

    grad_next = np.multiply(grad_prev, sigmoidGradient(output))

    return grad_next

def act_back(input, output, grad_prev, w, b):
    """ Args:
        output: a numpy array contains output data
        input: a numpy array contains input data layer, defined in testLeNet.m
        param: parameters, a cell array

        Returns:
        para_grad: a cell array stores gradients of parameters
        input_od: gradients w.r.t input data
    """
    w_grad = (input.T).dot(grad_prev)
    b_grad = np.sum(grad_prev, axis=0, keepdims=True)
    grad_next = grad_prev.dot(w.T)

    return w_grad, b_grad, grad_next


def update_param (param, param_grad, hyper_parameters):
    """Update the parameters with sgd with momentum

  Args:

  Returns:
    """
    lr = hyper_parameters['learning_rate']
    for key in param:
        param[key] = param[key] - (lr * param_grad[key])

    return param

def grad_calc (param, x, hyper_para):
    #x
    w1 = param['w1'] #784*100
    w2 = param['w2'] #100x784
    b1 = param['b1'] #1x100
    b2 = param['b2'] #1x784

    #### Dropout during training
    mask = np.random.binomial(1, 1 - hyper_para['drop_out'], x.shape)
    x = np.multiply(x, mask)

    #### Forward pass

    a1 = act_forward(x, w1, b1) #nx100 = nx784 * 784*100

    h1 = sigmoid_forward(a1) #n x 100 #same as before,

    a2 = act_forward(h1, w2, b2)    #nx100 = nx100 * 100x100

    x_hat = h2 = sigmoid_forward(a2) #n x 100 #same as before,

    ###### Backward Pass
    #act_back(input, output, grad_prev, w, b)
    #sigmoid_back (input, output, grad_prev ):

    d4 = x_hat - x  # nx10

    d3 = sigmoid_back(a2, h2, d4)

    [w2_grad, b2_grad, d2] = act_back(h1, a2, d3, w2, b2)

    d1 = sigmoid_back(a1, h1, d2)

    [w1_grad, b1_grad, d0] = act_back(x, a1, d1, w1, b1)    #784*100 = nx784' * nx100

    param_grad = {}
    param_grad['w1'] = w1_grad
    param_grad['w2'] = w2_grad
    param_grad['b1'] = b1_grad
    param_grad['b2'] = b2_grad

    return param_grad

def loss_calc(param, xtrain, ytrain, hyper_para):

    w1 = param['w1']  # 784*100
    w2 = param['w2']  # 100x784
    b1 = param['b1']  # 1x100
    b2 = param['b2']  # 1x784

    ###### Drop out during testing
    mask = np.ones((xtrain.shape)) * (1.0 - hyper_para['drop_out'])
    xtrain = np.multiply(xtrain, mask)

    #### Forward pass
    a1 = act_forward(xtrain, w1, b1)  # nx100 = nx784 * 784*100

    h1 = sigmoid_forward(a1)  # n x 100 #same as before,

    a2 = act_forward(h1, w2, b2)  # nx100 = nx100 * 100x100

    x_hat = sigmoid_forward(a2)  # n x 100 #same as before,

    loss = (xtrain * np.log(x_hat)) + ((1 - xtrain) * np.log(1 - x_hat))
    loss = -loss
    loss = np.sum(loss, axis=0)    #sum across all rows, examples
    loss = np.sum(loss, axis=0)     #sum across all cols, pixel values
    loss = loss / xtrain.shape[0]
    return loss

def plot_ce_train_valid (train_ce, valid_ce, hyper_para):
    date = time.strftime("%Y-%m-%d_%H_%M")
    plt.plot(train_ce)
    plt.plot(valid_ce)
    plt.title('AutoEncoder Error vs Epochs')
    a = 'Para:' + '\tBatch_size=' + str(hyper_para['batch_size'])
    b =  '\tlearning_rate=' + str(hyper_para['learning_rate'])
    c = '\tk=' + str(hyper_para['k'])
    d = '\tepochs: ' + str(hyper_para['epochs'])
    e = '\tDrop_out: ' + str(int(100*hyper_para['drop_out']))
    plt.suptitle(a +b + c + d + e)
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Error')
    plt.legend(['Train Cross Entropy', 'Valid Cross Entropy'], loc='upper right')
    plt.savefig('./figures/Q5_e' + date + '.png')
    plt.show()

def visualize (w1, hyper_para, show_flag, epoch) :
    date = time.strftime("%Y-%m-%d_%H_%M_%s")

    if w1.shape[0] == 784:
        nrow = 10
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
        a = 'Para:' + '\tBatch_size=' + str(hyper_para['batch_size'])
        b = '\tlearning_rate=' + str(hyper_para['learning_rate'])
        c = '\tk=' + str(hyper_para['k'])
        d = '\tepochs: ' + str(hyper_para['epochs'])
        if show_flag == 0:
            d = '\tepochs: ' + str(epoch)
        e = '\tDrop_out: ' + str(int(100 * hyper_para['drop_out']))
        plt.suptitle(a + b + c + d + e)
        plt.savefig('./figures/Q5_e_weights' + date + '.png')
        if show_flag == 1:
            plt.show()

def save_obj(obj, name, epoch ):
    with open('obj/'+ name + epoch +'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,epoch ):
    with open('obj/' + name + epoch + '.pkl', 'rb') as f:
        return pickle.load(f)
