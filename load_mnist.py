import numpy as np
from PIL import Image


def load_mnist ():
###Read train data
    xtr = []
    ytr = []
    f = open('./digitstrain.txt', 'r').readlines()
    N = len(f) - 1
    for i in range(0, N):
    #for i in range(0, 500):

        a = f[i].split(',')
        try:
            b = [float(num_string) for num_string in a]
        except ValueError, e:
            print "error", e, "on line", i, "line", f[i]
        ytr.append(int(b[len(b) - 1]))
        xtr.append(b[:-1])

    samples = len(xtr)
    size = len(xtr[0])
    xtrain = np.zeros((samples, size))

    for i in range(0, len(xtr)):
        xtrain[i, :] = np.array(xtr[i])
    ytrain = np.array(ytr)

    ###Read test data
    xtr = []
    ytr = []
    f = open('./digitstest.txt', 'r').readlines()
    N = len(f) - 1
    for i in range(0, N):
        a = f[i].split(',')
        try:
            b = [float(num_string) for num_string in a]
        except ValueError, e:
            print "error", e, "on line", i, "line", f[i]
        ytr.append(int(b[len(b) - 1]))
        xtr.append(b[:-1])

    samples = len(xtr)
    size = len(xtr[0])
    xtest = np.zeros((samples, size))

    for i in range(0, len(xtr)):
        xtest[i, :] = np.array(xtr[i])
    ytest = np.array(ytr)

###### Read Validation data

    xvalidate = []
    yvalidate = []


    xtr = []
    ytr = []
    f = open('./digitsvalid.txt', 'r').readlines()
    N = len(f) - 1
    for i in range(0, N):
        a = f[i].split(',')
        try:
            b = [float(num_string) for num_string in a]
        except ValueError, e:
            print "error", e, "on line", i, "line", f[i]
        ytr.append(int(b[len(b) - 1]))
        xtr.append(b[:-1])

    samples = len(xtr)
    size = len(xtr[0])
    xvalidate = np.zeros((samples, size))

    for i in range(0, len(xtr)):
        xvalidate[i, :] = np.array(xtr[i])
    yvalidate = np.array(ytr)

    return (xtrain, ytrain, xvalidate, yvalidate, xtest, ytest)