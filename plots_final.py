import matplotlib.pyplot as plt
import time

def plot_ce_train_valid (train_ce, valid_ce, hyper_para):
    date = time.strftime("%Y-%m-%d_%H_%M")
    plt.plot(train_ce)
    plt.plot(valid_ce)
    plt.title('Error vs Epochs')
    plt.suptitle('Para:' + '\tBatch_size=' + str(hyper_para['batch_size']) + '\tlearning_rate=' + str(hyper_para['learning_rate']) + '\tk=' + str(hyper_para['k']))
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Error')
    plt.legend(['Train Cross Entropy', 'Valid Cross Entropy'], loc='upper right')
    plt.show()
    plt.savefig('./figures/Q5_a' + date + '.png')


