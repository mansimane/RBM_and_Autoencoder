# p = np.reshape(w1, (280,280))
# m = Image.fromarray(p*150)
# m.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
def visualize (w1) :
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
        plt.show()
    else:
        nrow = 5
        ncol = 2

        fig = plt.figure(figsize=(ncol + 1, nrow + 1))

        gs = gridspec.GridSpec(nrow, ncol,
                               wspace=0.0, hspace=0.0,
                               top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                               left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
        cnt = 0
        for i in range(nrow):
            for j in range(ncol):
                im = np.reshape(w1[:, cnt], (10, 10))
                ax = plt.subplot(gs[i, j])
                ax.imshow(im)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                cnt = cnt + 1
        plt.title('Weights at layer 2')
        plt.show()