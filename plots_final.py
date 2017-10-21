import matplotlib.pyplot as plt
import time

def plot_ce_train_valid (train_ce, valid_ce, hyper_para):
    date = time.strftime("%Y-%m-%d_%H_%M")
    plt.plot(train_ce)
    plt.plot(valid_ce)
    plt.title('Error vs Epochs')
    a = 'Para:' + '\tBatch_size=' + str(hyper_para['batch_size'])
    b = '\tlearning_rate=' + str(hyper_para['learning_rate'])
    c = '\tk=' + str(hyper_para['k'])
    plt.suptitle(a  + b + c)
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Error')
    plt.legend(['Train Cross Entropy', 'Valid Cross Entropy'], loc='upper right')
    plt.show()
    hh = hyper_para['']
    plt.savefig('./figures/Q5_g_rbm' + date + '.png')
#    plt.close()


