# from PIL import Image
# from load_mnist import *
# from functions_final import *
# import matplotlib.pyplot as plt
# import random
# from plots_final import *
# from visual import *
# import time
#
# #Setting parameters
# input_layer_size  = 784  # 28x28 Input Images of Digits
# hidden_layer_size = 100   # 100 hidden units
# num_labels = 10          # 10 labels, from 1 to 10
#                           # (note that we have mapped "0" to label 10)
#
# # Traning Parameters
# hyper_para = {}
# hyper_para['batch_size'] = 16
# hyper_para['epochs'] = 100
# hyper_para['w_rate'] = 0.001        #Learning rate for w
# hyper_para['b_rate'] = 0.001         #Learning rate for b
# hyper_para['llambda'] = 0.2          #Learning rate for b
# hyper_para['mu'] = 0.0            #Momentum
# hyper_para['decay'] = 0.0005     #weight decay
# hyper_para['random_seed'] = 0
# hyper_para['no_of_h_layer'] = 1
# hyper_para['wt_intial_mean'] = 0
# hyper_para['wt_intial_sigma'] = 0.1
#
#
# # Load Training Data
# xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist()
# no_of_train_samples = len(ytrain)
#
# #Unroll parameters and randomly initialize them
# mu = hyper_para['wt_intial_mean']
# sigma = hyper_para['wt_intial_sigma'] # mean and standard deviation
#
# # Random seed for the run
# random_seed = hyper_para['random_seed']
#
# np.random.seed(random_seed)
#
# w1 = np.random.normal(mu, sigma, (input_layer_size * hidden_layer_size))
# w1 = w1.reshape(input_layer_size, hidden_layer_size)
#
# w2 = np.random.normal(mu, sigma, (hidden_layer_size * num_labels))
# w2 = w2.reshape(hidden_layer_size,num_labels)
#
# b1 = np.zeros((1, hidden_layer_size))
# b2 = np.zeros((1, num_labels))
#
# param = [[w1,b1], [w2,b2]]
# param_winc = copy.deepcopy(param)
# for i in range(len(param_winc)):
#     param_winc[i][0] = param_winc[i][0] * 0
#     param_winc[i][1] = param_winc[i][1] * 0
#
#
# display_interval = 9
# snapshot = 5000
# no_of_train_samples = len(ytrain)
#
# # Variables Storing Results
# J_train = 0
# J_valid = 0
# train_ce = []   #Cross Entropy = CE
# valid_ce = []
# test_ce = []
# train_mce = []   #Mean Classification Error
# valid_mce = []
#
# test_mce = []
# # learning iterations
# indices = range(no_of_train_samples)
# random.shuffle(indices)
# batch_size = hyper_para['batch_size']
# epochs = hyper_para['epochs']
# max_iter = no_of_train_samples / batch_size       #Number of batches in each epoch
#
# train_accuracy = 0
# start_time = time.clock()
# for epoch in range(epochs):
#     for step in range(max_iter):
#         # get mini-batch and
#         start_idx = step * batch_size % no_of_train_samples
#         end_idx = (step + 1) * batch_size % no_of_train_samples
#
#         if start_idx > end_idx:
#             random.shuffle(indices)
#             continue
#         idx = indices[start_idx: end_idx]
#
#         param_grad = grad_calc(param, xtrain[idx, :], ytrain[idx])
#         [param, param_winc] = update_param(param, param_grad, param_winc, hyper_para)
#
#     [train_error, J_train] = calc_accuracy(param, xtrain, ytrain)
#     [valid_error, J_valid] = calc_accuracy(param, xvalidate, yvalidate)
#     [test_error, J_test] = calc_accuracy(param, xtest, ytest)
#     print 'Current time', time.clock(), 'epoch', epoch, '\tTrain CE', J_train, '\tPA', ((1-train_error)*100), '\t\tValid CE', J_valid, '\tPA', ((1 - valid_error)*100), '\tTest PA', ((1-test_error)*100), '\t\ttest CE', J_test
#     train_ce.append(J_train)
#     valid_ce.append(J_valid)
#     test_ce.append(J_test)
#     train_mce.append(train_error)
#     valid_mce.append(valid_error)
#
# end_time = time.clock()
# print "time taken\t", end_time - start_time
# w1 = param[0][0]
# w2 = param[1][0]
# plot_ce_train_valid(train_ce, valid_ce, hyper_para)
# #plot_ce_mce_train_valid(train_ce, valid_ce, train_mce, valid_mce, hyper_para)
# visualize(w1)
# visualize(w2)
#
from PIL import Image
from load_mnist import *
from functions_final import *
import matplotlib.pyplot as plt
import random
from plots_final import *
from visual import *
import time

#Setting parameters
input_layer_size  = 784  # 28x28 Input Images of Digits
hidden_layer_size = 100   # 100 hidden units
num_labels = 10          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

# Traning Parameters
hyper_para = {}
hyper_para['batch_size'] = 16
hyper_para['epochs'] = 100
hyper_para['w_rate'] = 0.1        #Learning rate for w
hyper_para['b_rate'] = 0.1         #Learning rate for b
hyper_para['llambda'] = 0.2          #Learning rate for b
hyper_para['mu'] = 0.0            #Momentum
hyper_para['decay'] = 0.0005     #weight decay
hyper_para['random_seed'] = 0
hyper_para['no_of_h_layer'] = 1
hyper_para['wt_intial_mean'] = 0
hyper_para['wt_intial_sigma'] = 0.1


# Load Training Data
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist()
no_of_train_samples = len(ytrain)

#Unroll parameters and randomly initialize them
mu = hyper_para['wt_intial_mean']
sigma = hyper_para['wt_intial_sigma'] # mean and standard deviation

# Random seed for the run
random_seed = hyper_para['random_seed']

np.random.seed(random_seed)

w1 = np.random.normal(mu, sigma, (input_layer_size * hidden_layer_size))
w1 = w1.reshape(input_layer_size, hidden_layer_size)

w2 = np.random.normal(mu, sigma, (hidden_layer_size * num_labels))
w2 = w2.reshape(hidden_layer_size,num_labels)

b1 = np.zeros((1, hidden_layer_size))
b2 = np.zeros((1, num_labels))

param = [[w1,b1], [w2,b2]]
param_winc = copy.deepcopy(param)
for i in range(len(param_winc)):
    param_winc[i][0] = param_winc[i][0] * 0
    param_winc[i][1] = param_winc[i][1] * 0


no_of_train_samples = len(ytrain)

# Variables Storing Results
J_train = 0
J_valid = 0
train_ce = []   #Cross Entropy = CE
valid_ce = []
test_ce = []
train_mce = []   #Mean Classification Error
valid_mce = []

test_mce = []
# learning iterations
indices = range(no_of_train_samples)
random.shuffle(indices)
batch_size = hyper_para['batch_size']
epochs = hyper_para['epochs']
max_iter = no_of_train_samples / batch_size       #Number of batches in each epoch

train_accuracy = 0
start_time = time.clock()
for epoch in range(epochs):
    for step in range(max_iter):
        # get mini-batch and
        start_idx = step * batch_size % no_of_train_samples
        end_idx = (step + 1) * batch_size % no_of_train_samples

        if start_idx > end_idx:
            random.shuffle(indices)
            continue
        idx = indices[start_idx: end_idx]

        param_grad = grad_calc(param, xtrain[idx, :], ytrain[idx])
        [param, param_winc] = update_param(param, param_grad, param_winc, hyper_para)

    [train_error, J_train] = calc_accuracy(param, xtrain, ytrain)
    [valid_error, J_valid] = calc_accuracy(param, xvalidate, yvalidate)
    [test_error, J_test] = calc_accuracy(param, xtest, ytest)
    print 'Current time', time.clock(), 'epoch', epoch, '\tTrain CE', J_train, '\tPA', ((1-train_error)*100), '\t\tValid CE', J_valid, '\tPA', ((1 - valid_error)*100), '\tTest PA', ((1-test_error)*100), '\t\ttest CE', J_test
    train_ce.append(J_train)
    valid_ce.append(J_valid)
    test_ce.append(J_test)
    train_mce.append(train_error)
    valid_mce.append(valid_error)

end_time = time.clock()
print "time taken\t", end_time - start_time
w1 = param[0][0]
w2 = param[1][0]
plot_ce_train_valid(train_ce, valid_ce, hyper_para)
#plot_ce_mce_train_valid(train_ce, valid_ce, train_mce, valid_mce, hyper_para)
visualize(w1)
visualize(w2)