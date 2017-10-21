import numpy as np
import copy

def relu_forward(z):
    g = np.maximum(z, 0, z)
    return g

def relu_back(input, output, grad_prev):
    grad_next = grad_prev * output
    return grad_next

def tanh_forward(z):
    g = np.tanh(z)
    return g

def tanh_back(input, output, grad_prev):
    grad_next = grad_prev * (1 - np.power(output, 2))
    return grad_next


def softmax_forward(a):

    if len(a.shape) == 1:   #if vector
        y = np.exp(a)
        y = y/np.sum(y, keepdims=True)
        assert (len(y) == len(a)), 'Y returned from softmax not match with input array'

    #a = samples x num_of_labels
    else:
        y = np.exp(a)
        y = y / np.sum(y, axis=1, keepdims=True)
    return y

#Currently will work only if a is vector
def soft_max_back_prop (a, idx):


    if len(a.shape) == 1:   #if vector
        y = np.ones((a.shape[0], 1))
        for i in range (0, len(a)):
            if (i == idx):
                y[i] = (a[i]*(1-a[i]))
            else:
                y[i] = a[i]*a[idx]
        assert (len(y) == len(a)), 'Y returned from softmax backprop not match with input array'

    #a = samples x num_of_labels
    else:
        n = a.shape[0]
        y = np.zeros((a.shape[0], a.shape[1]))
        for i in range(0, n):
            max_idx = a[i, :].argmax(axis=0)
            for j in range (0,a.shape[1]):
                if (j == max_idx):
                    y[i, j] = (a[i, j]*(1-a[i, j]))
                else:
                    y[i, j] = (a[i, j] * (1 - a[i, j]))
        assert (len(y) == a.shape[0]), 'Y returned from softmax not match with input matrix'

    return y

def calc_accuracy (param, x, y):

    #x
    w1 = copy.deepcopy(param[0][0]) #784*100
    w2 = copy.deepcopy(param[1][0]) #100x10
    b1 = copy.deepcopy(param[0][1]) #1x100
    b2 = copy.deepcopy(param[1][1]) #1x10


    no_of_samples = len(y)
    no_of_layers = len(param)

    a1 = act_forward(x, w1, b1)  #nx100 = nx784 * 784*100
    h1 = sigmoid_forward(a1) #n x 100 #same as before, bias added to every row ans col

    #Activation2
    a2 = act_forward(h1, w2, b2)    #nx10 = nx100 * 100x10

    #Softmax
    y_pred = softmax_forward(a2)     # nx10
    y_pred_digit = y_pred.argmax(axis=1)
    error = len(np.where(y != y_pred_digit)[0]) / float(len(y))

    y_prob_right = y_pred[range(no_of_samples), y]
    y_prob_right = -np.log(y_prob_right)

    J = np.sum(y_prob_right)/len(y)

    return error, J

def sigmoidGradient (z):
    g = np.multiply(z , (1-z))

    return g

def sigmoid_forward (z):
    g = 1.0/(1.0 + np.exp(-z))
    return g

""" Activation
"""

def act_forward(input, w, b):
    """ Args:

        Returns:
    """
    a = input.dot(w) + b
    return a

def softmax_back (y_pred, y_correct):
    output = y_pred - y_correct
    return output

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

def sigmoid_back (input, output, grad_prev ):

    grad_next = np.multiply(grad_prev, sigmoidGradient(output))

    return grad_next

def calc_accuracy_2layer (param, x, y):

    #x
    w1 = copy.deepcopy(param[0][0]) #784*100
    w2 = copy.deepcopy(param[1][0]) #100x100
    w3 = copy.deepcopy(param[2][0]) #100x10

    b1 = copy.deepcopy(param[0][1]) #1x100
    b2 = copy.deepcopy(param[1][1]) #1x100
    b3 = copy.deepcopy(param[2][1]) #1x10


    no_of_samples = len(y)
    no_of_layers = len(param)

    a1 = act_forward(x, w1, b1) #nx100 = nx784 * 784*100

    h1 = sigmoid_forward(a1) #n x 100 #same as before,

    a2 = act_forward(h1, w2, b2)    #nx100 = nx100 * 100x100

    h2 = sigmoid_forward(a2) #n x 100 #same as before,

    a3 = act_forward(h2, w3, b3)    #nx10 = nx100 * 100x10

    y_pred = softmax_forward(a3)     # nx10

    y_pred_digit = y_pred.argmax(axis=1)
    error = len(np.where(y != y_pred_digit)[0]) / float(len(y))

    y_prob_right = y_pred[range(no_of_samples), y]
    y_prob_right = -np.log(y_prob_right)

    J = np.sum(y_prob_right)/len(y)

    return error, J

def batch_norm_forward (x, gamma, beta, eps):
    no_of_samples = x.shape[0]

    mean = np.sum(x, axis=0) / no_of_samples

    x_min_mu = x - mean

    var = (np.sum((x_min_mu ** 2), axis=0)) / float(no_of_samples)

    sigma = np.sqrt(var + eps)

    one_by_var = 1.0 / sigma #Element-wise as #

    x_norm = x_min_mu * one_by_var

    y = (gamma * x_norm) + beta

    batch_data_para = []
    batch_data_para.append(x_norm)
    batch_data_para.append(x_min_mu)
    batch_data_para.append(sigma)
    batch_data_para.append(var)

    return y, batch_data_para

def batch_norm_back (input, output, grad_prev, gamma, beta, batch_data_para, eps):

    x_norm = batch_data_para[0]
    x_min_mu = batch_data_para[1]
    sigma = batch_data_para[2]
    var = batch_data_para[3]


    no_of_samples = grad_prev.shape[0]

    beta_grad = np.sum(grad_prev, axis=0)

    gamma_grad = np.sum(grad_prev*x_norm, axis=0)

    xnorm_grad = grad_prev * gamma

    var_grad = np.sum(xnorm_grad * x_min_mu, axis=0)

    x_grad = xnorm_grad / sigma #**

    sqrt_var_grad = -(1.0 / (sigma ** 2)) * var_grad

    var_grad2 = (0.5/ np.sqrt(var + eps)) *sqrt_var_grad

    x_min_mu_grad = (1.0/no_of_samples) * (np.ones((input.shape[0], input.shape[1])) * var_grad2)

    x_min_mu_grad2 = x_min_mu_grad * 2 * x_min_mu

    x_grad_1 = x_min_mu_grad2 + x_grad

    mu_grad = -1.0 * np.sum(x_min_mu_grad2 + x_grad, axis=0)

    x_grad2 = np.ones((input.shape[0], input.shape[1])) * mu_grad

    x_grad2 =  x_grad2 /no_of_samples

    grad_next = x_grad_1 + x_grad2

    return gamma_grad, beta_grad, grad_next

def calc_accuracy_2layer_batch_norm (param, x, y, eps):

    w1 =        copy.deepcopy(param[0][0]) #784*100
    gamma1 =    copy.deepcopy(param[1][0]) #1x100
    w2 =        copy.deepcopy(param[2][0]) #100x100
    gamma2=     copy.deepcopy(param[3][0]) #1x100
    w3 =        copy.deepcopy(param[4][0]) #100x10

    b1 =        copy.deepcopy(param[0][1]) #1x100
    beta1 =     copy.deepcopy(param[1][1]) #1x100
    b2 =        copy.deepcopy(param[2][1]) #1x100
    beta2 =     copy.deepcopy(param[3][1]) #1x100
    b3 =        copy.deepcopy(param[4][1]) #1x10

    no_of_samples = len(y)
    #### Forward pass

    a1 = act_forward(x, w1, b1) #nx100 = nx784 * 784*100

    h1 = sigmoid_forward(a1) #n x 100 #same as before,

    r1, batch_data_para1 = batch_norm_forward(h1, gamma1, beta1, eps) #n x 100 #same as before,

    a2 = act_forward(r1, w2, b2)    #nx100 = nx100 * 100x100

    h2 = sigmoid_forward(a2) #n x 100 #same as before,

    r2, batch_data_para2 = batch_norm_forward(h2, gamma2, beta2, eps) #n x 100 #same as before,

    a3 = act_forward(r2, w3, b3)    #nx10 = nx100 * 100x10

    y_pred = softmax_forward(a3)     # nx10

    y_pred_digit = y_pred.argmax(axis=1)
    error = len(np.where(y != y_pred_digit)[0]) / float(len(y))

    y_prob_right = y_pred[range(no_of_samples), y]
    y_prob_right = -np.log(y_prob_right)

    J = np.sum(y_prob_right)/len(y)

    return error, J

def calc_accuracy_2layer_batch_norm_relu (param, x, y, eps, hyper_para):

    w1 =        copy.deepcopy(param[0][0]) #784*100
    gamma1 =    copy.deepcopy(param[1][0]) #1x100
    w2 =        copy.deepcopy(param[2][0]) #100x100
    gamma2=     copy.deepcopy(param[3][0]) #1x100
    w3 =        copy.deepcopy(param[4][0]) #100x10

    b1 =        copy.deepcopy(param[0][1]) #1x100
    beta1 =     copy.deepcopy(param[1][1]) #1x100
    b2 =        copy.deepcopy(param[2][1]) #1x100
    beta2 =     copy.deepcopy(param[3][1]) #1x100
    b3 =        copy.deepcopy(param[4][1]) #1x10

    no_of_samples = len(y)
    #### Forward pass

    a1 = act_forward(x, w1, b1) #nx100 = nx784 * 784*100

    if hyper_para['non_linearity'] == 'relu':
        h1 = relu_forward(a1) #n x 100 #same as before,
    elif hyper_para['non_linearity'] == 'tanh':
        h1 = tanh_forward(a1)  # n x 100 #same as before,

    r1, batch_data_para1 = batch_norm_forward(h1, gamma1, beta1, eps) #n x 100 #same as before,

    a2 = act_forward(r1, w2, b2)    #nx100 = nx100 * 100x100

    if hyper_para['non_linearity'] == 'relu':
        h2 = relu_forward(a2) #n x 100 #same as before,
    elif hyper_para['non_linearity'] == 'tanh':
        h2 = tanh_forward(a1)

    r2, batch_data_para2 = batch_norm_forward(h2, gamma2, beta2, eps) #n x 100 #same as before,

    a3 = act_forward(r2, w3, b3)    #nx10 = nx100 * 100x10

    y_pred = softmax_forward(a3)     # nx10

    y_pred_digit = y_pred.argmax(axis=1)
    error = len(np.where(y != y_pred_digit)[0]) / float(len(y))

    y_prob_right = y_pred[range(no_of_samples), y]
    y_prob_right = -np.log(y_prob_right)

    J = np.sum(y_prob_right)/len(y)

    return error, J
