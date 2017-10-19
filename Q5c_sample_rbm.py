from functions_final import *
from config import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time

mu = hyper_para['w_init_mu']
sigma = hyper_para['w_init_sig']

epoch = 150
param = load_obj('param', str(epoch))


# override number of steps in CD with 100
hyper_para['k'] = 1000
no_of_gibbs_chains = 100

x = np.random.normal(mu, sigma, (no_of_gibbs_chains * input_layer_size))
x = x.reshape(no_of_gibbs_chains, input_layer_size)

x_p, h_p = gibbs_step(param, x, hyper_para)

date = time.strftime("%Y-%m-%d_%H_%M")

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
        im = np.reshape(x_p[cnt, :], (28, 28))
        ax = plt.subplot(gs[i, j])
        ax.imshow(im)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        cnt = cnt + 1

#plt.title('Digits generated with RBM')
plt.suptitle('Para:' + '\tBatch_size=' + str(hyper_para['batch_size']) + '\tlearning_rate=' + str(
    hyper_para['learning_rate']) + '\tk=' + str(hyper_para['k']))
plt.savefig('./figures/Q5_c_numbers' + date + '.png')
plt.show()

