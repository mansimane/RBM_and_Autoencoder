#Setting parameters
                          # (note that we have mapped "0" to label 10)

# Traning Parameters
hyper_para = {}
hyper_para['batch_size'] = 1
hyper_para['epochs'] = 120

hyper_para['learning_rate'] = 0.01        #Learning rate for w

hyper_para['mu'] = 0.5            #Momentum
hyper_para['decay'] = 0.0005      #weight decay
hyper_para['random_seed'] = 0
hyper_para['eps'] = 0.001
hyper_para['k'] = 10     #Gibbs sampling steps

#Unroll parameters and randomly initialize them
hyper_para['w_init_mu'] = 0
hyper_para['w_init_sig'] = 0.1 # mean and standard deviation
hyper_para['display_interval'] = 9
hyper_para['snapshot'] = 5000

#
hyper_para['no_of_h_layer'] = 1

hyper_para['input_layer_size']  = 784  # 28x28 Input Images of Digits
input_layer_size = hyper_para['input_layer_size']

hyper_para['hidden_layer_1_size'] = 100   # 100 hidden units
hidden_layer_1_size = hyper_para['hidden_layer_1_size']

hyper_para['hidden_layer_2_size'] = 100   # 100 hidden units
hyper_para['num_labels'] = 10          # 10 labels, from 1 to 10
