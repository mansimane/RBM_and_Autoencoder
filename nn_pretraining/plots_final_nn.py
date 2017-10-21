import matplotlib.pyplot as plt
import time

def plot_ce_train_valid (train_ce, valid_ce, hyper_para):
    date = time.strftime("%Y-%m-%d_%H_%M")
    plt.plot(train_ce)
    plt.plot(valid_ce)
    plt.title('Train Error vs Epochs')
    c = 'Pretrain:' + str(hyper_para['pretrain'])
    plt.suptitle(c +'Para:' + '\tBatch_size=' + str(hyper_para['batch_size']) + '\tw_rate=' + str(
        hyper_para['w_rate']) + '\tb_rate='
                 + str(hyper_para['b_rate']) + '\tmu=' + str(hyper_para['mu']) + '\tdecay=' + str(
        hyper_para['decay']) + '\tRandom Seed=' + str(hyper_para['random_seed']))
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Error')
    plt.legend(['Train Cross Entropy', 'Valid Cross Entropy'], loc='upper right')
    plt.savefig('./figures/Q6_a' + date + '.png')
    plt.show()
    #plt.savefig('./figures/Q6_a' + '_' + 'B' + str(hyper_para['batch_size']) + '_wr' + str(hyper_para['w_rate'])+  '_br'+ str(hyper_para['b_rate']) + '_mu' + str(hyper_para['mu']) + '_d' + str(hyper_para['decay']) + '_rs' + str(hyper_para['random_seed'])) + date + '.png'

