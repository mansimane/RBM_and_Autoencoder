from small_functions_final import *
import math
import copy

def grad_calc(param, x, y):
    #x
    w1 = copy.deepcopy(param[0][0]) #784*100
    w2 = copy.deepcopy(param[1][0]) #100x10
    b1 = copy.deepcopy(param[0][1]) #1x100
    b2 = copy.deepcopy(param[1][1]) #1x10

    no_of_samples = len(y)
    #### Forward pass

    a1 = act_forward(x, w1, b1) #nx100 = nx784 * 784*100

    h1 = sigmoid_forward(a1) #n x 100 #same as before, bias added to every row ans col

    #Activation2
    a2 = act_forward(h1, w2, b2)    #nx10 = nx100 * 100x10
    y_pred = softmax_forward(a2)     # nx10

    ###### Backward Pass

    y_correct = 0 * y_pred # nx10
    y_correct[range(no_of_samples), y] = 1.0 # nx10
    d3 = softmax_back(y_pred, y_correct)  # nx10

    [w2_grad, b2_grad, d2] = act_back(h1, a2, d3, w2, b2)

    d1 = sigmoid_back(a1, h1, d2)

    [w1_grad, b1_grad, d0] = act_back(x, a1, d1, w1, b1)    #784*100 = nx784' * nx100
    param_grad = [[w1_grad, b1_grad], [w2_grad, b2_grad]]


    return param_grad

def update_param (param, param_grad, param_winc, hyper_parameters):
    """Update the parameters with sgd with momentum

  Args:
    w_rate (scalar): sgd rate for updating w
    b_rate (scalar): sgd rate for updating b
    mu (scalar): momentum
    decay (scalar): weight decay of w
    params (list): original weight parameters
    param_winc (list): buffer to store history gradient accumulation
    param_grad (list): gradient of parameter

  Returns:
    params_ (list): updated parameters
    param_winc_ (list): gradient buffer of previous step
    """

    params_loc = copy.deepcopy(param)
    param_winc_loc = copy.deepcopy(param_winc)

  # TODO: your implementation goes below this comment
  # implementation begins
    w_rate = hyper_parameters['w_rate']
    b_rate = hyper_parameters['b_rate']
    decay = hyper_parameters['decay']
    mu = hyper_parameters['mu']
    for i in range(0, len(param)):
         param_winc_loc[i][0] = mu*param_winc[i][0] + w_rate*param_grad[i][0] + decay*param[i][0]
         param_winc_loc[i][1] = mu*param_winc[i][1] + b_rate*param_grad[i][1] + decay*param[i][1]

         params_loc[i][0] -= param_winc_loc[i][0]
         params_loc[i][1] -= param_winc_loc[i][1]

        #params_loc[i][0] -= w_rate * (param_grad[i][0] + decay * param[i][0])
        #params_loc[i][1] -= b_rate * (param_grad[i][1] + decay * param[i][1])
        #params_loc[i][0] -= w_rate * (param_grad[i][0])
        #params_loc[i][1] -= b_rate * (param_grad[i][1])



        # implementation ends
    assert ((len(params_loc) == len(param_grad))), 'params does not have the right length'
    assert ((len(param_winc_loc) == len(param_grad))), 'param_winc does not have the right length'


    return params_loc, param_winc_loc

def grad_calc_2layer (param, x, y):
    #x
    w1 = copy.deepcopy(param[0][0]) #784*100
    w2 = copy.deepcopy(param[1][0]) #100x100
    w3 = copy.deepcopy(param[2][0]) #100x10
    b1 = copy.deepcopy(param[0][1]) #1x100
    b2 = copy.deepcopy(param[1][1]) #1x100
    b3 = copy.deepcopy(param[2][1]) #1x10

    no_of_samples = len(y)
    #### Forward pass

    a1 = act_forward(x, w1, b1) #nx100 = nx784 * 784*100

    h1 = sigmoid_forward(a1) #n x 100 #same as before,

    a2 = act_forward(h1, w2, b2)    #nx100 = nx100 * 100x100

    h2 = sigmoid_forward(a2) #n x 100 #same as before,

    a3 = act_forward(h2, w3, b3)    #nx10 = nx100 * 100x10

    y_pred = softmax_forward(a3)     # nx10

    ###### Backward Pass

    y_correct = 0 * y_pred # nx10
    y_correct[range(no_of_samples), y] = 1.0 # nx10

    d4 = softmax_back(y_pred, y_correct)  # nx10

    [w3_grad, b3_grad, d3] = act_back(h2, a3, d4, w3, b3)

    d2 = sigmoid_back(a2, h2, d3)

    [w2_grad, b2_grad, d1] = act_back(h1, a2, d2, w2, b2)    #784*100 = nx784' * nx100

    d1 = sigmoid_back(a1, h1, d1)

    [w1_grad, b1_grad, d0] = act_back(x, a1, d1, w1, b1)    #784*100 = nx784' * nx100

    param_grad = [[w1_grad, b1_grad], [w2_grad, b2_grad], [w3_grad, b3_grad]]

    return param_grad

def grad_calc_2layer_batch_norm (param, x, y, eps):
    #x
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

    ###### Backward Pass

    y_correct = 0 * y_pred # nx10
    y_correct[range(no_of_samples), y] = 1.0 # nx10

    d6 = softmax_back(y_pred, y_correct)  # nx10

    w3_grad, b3_grad, d5 = act_back(r2, a3, d6, w3, b3)

    [gamma2_grad, beta2_grad, d4] = batch_norm_back(h2, r2, d5, gamma2, beta2, batch_data_para2)    #batch_norm_back (input, output, grad_prev, batch_data_para):

    #Sigmoid 2
    d3 = sigmoid_back(a2, h2, d4)

    w2_grad, b2_grad, d1 = act_back(r1, a2, d3, w2, b2)    #784*100 = nx784' * nx100

    [gamma1_grad, beta1_grad, d2] = batch_norm_back(h1, r1, d3,gamma1, beta1, batch_data_para1)    #batch_norm_back (input, output, grad_prev, batch_data_para):


    d1 = sigmoid_back(a1, h1, d2)

    w1_grad, b1_grad, d0 = act_back(x, a1, d1, w1, b1)    #784*100 = nx784' * nx100

    param_grad = [[w1_grad, b1_grad], [gamma1_grad, beta1_grad], [w2_grad,b2_grad], [gamma2_grad,beta2_grad], [w3_grad, b3_grad] ]
    #param_grad = []
    #print gamma1_grad, gamma2_grad, "beta1_grad", beta1_grad, beta2_grad
    #print "beta1_grad", beta1_grad

    return param_grad

def grad_calc_2layer_batch_norm_relu (param, x, y, eps,hyper_para):
    #x
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

    ###### Backward Pass

    y_correct = 0 * y_pred # nx10
    y_correct[range(no_of_samples), y] = 1.0 # nx10

    d6 = softmax_back(y_pred, y_correct)  # nx10

    w3_grad, b3_grad, d5 = act_back(r2, a3, d6, w3, b3)

    [gamma2_grad, beta2_grad, d4] = batch_norm_back(h2, r2, d5, gamma2, beta2, batch_data_para2)    #batch_norm_back (input, output, grad_prev, batch_data_para):

    #Sigmoid 2

    if hyper_para['non_linearity'] == 'relu':
        d3 = relu_back(a2, h2, d4)
    elif hyper_para['non_linearity'] == 'tanh':
        d3 = tanh_back(a2, h2, d4)

    w2_grad, b2_grad, d1 = act_back(r1, a2, d3, w2, b2)    #784*100 = nx784' * nx100

    [gamma1_grad, beta1_grad, d2] = batch_norm_back(h1, r1, d3,gamma1, beta1, batch_data_para1)    #batch_norm_back (input, output, grad_prev, batch_data_para):

    if hyper_para['non_linearity'] == 'relu':
        d1 = relu_back(a1, h1, d2)
    elif hyper_para['non_linearity'] == 'tanh':
        d1 = tanh_back(a1, h1, d2)

    w1_grad, b1_grad, d0 = act_back(x, a1, d1, w1, b1)    #784*100 = nx784' * nx100

    param_grad = [[w1_grad, b1_grad], [gamma1_grad, beta1_grad], [w2_grad,b2_grad], [gamma2_grad,beta2_grad], [w3_grad, b3_grad] ]
    #param_grad = []
    #print gamma1_grad, gamma2_grad, "beta1_grad", beta1_grad, beta2_grad
    #print "beta1_grad", beta1_grad

    return param_grad
