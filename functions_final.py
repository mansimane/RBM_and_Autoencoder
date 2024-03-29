import numpy as np
import math
import copy
import pickle

def sigmoid_forward (z):
    g = 1.0/(1.0 + np.exp(-z))
    return g

def h_calc(param, x_bin, hyper_para):
    #x  nx784
    n = np.shape(x_bin)[0]
    no_of_hidden_units = hyper_para['hidden_layer_1_size']
    w = param['w']                       #w =               #784*100
    h = x_bin.dot(w) + param['b']   #nx100 = nx784 * 784*100
    h_p = sigmoid_forward(h)    #*** what to update and return param or h ?
    h_bin = h_p > np.random.rand(n, no_of_hidden_units)

    return h_p,h_bin      #row col check?

def x_calc(param, h_bin, hyper_para):
    w = param['w']
    t = param['c'] + h_bin.dot(w.T)    #1x784 = 1x784 + (1x100 * 100x784)
    x_p = sigmoid_forward(t)
    x_bin = x_p > np.random.rand(x_p.shape[0], x_p.shape[1])  #*** wrong

    return x_p, x_bin

def gibbs_step(param, xtrain, hyper_para):
    k = hyper_para['k']
    x_p = xtrain
    x_bin = x_p
    for i in range(k):
        h_p, h_bin = h_calc(param, x_bin, hyper_para)   #what h or parm should be collected??
        x_p, x_bin = x_calc(param, h_bin, hyper_para)   #

    return x_p, h_p

def update_param (param, x_p, xtrain, hyper_para):
    """Update the parameters with sgd with momentum
  Args:
  Returns:

    """
    params_loc = copy.deepcopy(param)

  # TODO: your implementation goes below this comment
    lr = hyper_para['learning_rate']
    x_neg = x_p
    x = xtrain

    h, h_bin = h_calc(param, x, hyper_para)
    h_neg, h_neg_bin = h_calc(param, x_neg, hyper_para)

    n = np.shape(xtrain)[0]

    w_grad = (x.T).dot(h) - (x_neg.T).dot(h_neg) #**divide by n?
    w_grad = w_grad/n
    param['w'] = param['w'] + (lr * w_grad)

    b_grad = np.sum(h - h_neg, axis = 0)
    b_grad = b_grad/n
    param['b'] = param['b'] + (lr * (b_grad))

    c_grad = np.sum(x - x_neg, axis=0)
    c_grad = c_grad/n
    param['c'] = param['c'] + (lr * c_grad)
        #param_winc_loc[i][0] = mu*param_winc[i][0] + w_rate*param_grad[i][0]
        #param_winc_loc[i][1] = mu*param_winc[i][1] + b_rate*param_grad[i][1]

        #params_loc[i][0] -= param_winc_loc[i][0]
        #params_loc[i][1] -= param_winc_loc[i][1]

        #params_loc[i][0] -= w_rate * (param_grad[i][0] + decay * param[i][0])
        #params_loc[i][1] -= b_rate * (param_grad[i][1] + decay * param[i][1])
        #params_loc[i][0] -= w_rate * (param_grad[i][0])
        #params_loc[i][1] -= b_rate * (param_grad[i][1])



        # implementation ends
#    assert ((len(param_winc_loc) == len(param_grad))), 'param_winc does not have the right length'
    return param



def loss_calc(param, xtrain, ytrain, hyper_para):

    h_p, h_bin = h_calc(param, xtrain, hyper_para)
    x_p, x_bin = x_calc(param, h_p, hyper_para)
    #xtrain = xtrain > 0.5
    loss = (xtrain * np.log(x_p)) + ((1 - xtrain) * np.log(1 - x_p))
    loss = -loss
    loss = np.sum(loss, axis=0)    #sum across all rows, examples
    loss = np.sum(loss, axis=0)     #sum across all cols, pixel values
    loss = loss / xtrain.shape[0]
    return loss

def save_obj(obj, name, epoch ):
    with open('obj/'+ name + epoch +'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,epoch ):
    with open('obj/' + name + epoch + '.pkl', 'rb') as f:
        return pickle.load(f)
