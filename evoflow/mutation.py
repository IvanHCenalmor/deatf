import numpy as np
from evoflow.network import initializations, activations
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU

class Mutation:
    
    def __init__(self, ev_hypers, max_lay, batch_normalization, drop, individual):
        self.ev_hypers = ev_hypers
        self.max_lay = max_lay
        self.batch_normalization = batch_normalization
        self.drop = drop
        self.individual = individual
        
    def apply_random_mutation(self, network, custom_mutation_list):
        
        if not custom_mutation_list:
            custom_mutation_list = self.methods()
            
        type_mutation = np.random.choice(custom_mutation_list)
        
        self.apply_mutation(type_mutation, network)
        
    def apply_mutation(self, mutation, network):
        
        if mutation in self.methods():
            eval('self.' + mutation + '(network)')
        else:
            raise ValueError('The mutation {} is not defined'.format(mutation))
        
    def methods(self):
        
        method_list = [method for method in dir(self.__class__) if not method.startswith('__')]
        
        method_list.remove('methods')
        method_list.remove('apply_mutation')
        method_list.remove('apply_random_mutation')
        
        
        if len(self.ev_hypers) == 0:
            method_list.remove('mut_hyper')
        if self.batch_normalization:
            method_list.remove('mut_batch_norm')
        if self.drop:
            method_list.remove('mut_dropout')
        
        return method_list
       
    def mut_weight_init(self, network):             # We change weight initialization function in all layers
        layer_pos = np.random.randint(network.number_hidden_layers)
        init_w_function = initializations[np.random.randint(len(initializations))]
        network.change_weight_init(layer_pos, init_w_function)


    def mut_activation(self, network):             # We change the activation function in layer
        layer_pos = np.random.randint(network.number_hidden_layers)
        init_a_function = activations[np.random.randint(len(activations))]
        network.change_activation(layer_pos, init_a_function)
        
    def mut_dropout(self, network):
        network.change_dropout()
        
    def mut_dropout_prob(self, network):
        network.change_dropout_prob()

    def mut_batch_norm(self, network):
        network.change_batch_norm() 
        
    def mut_hyper(self, network):                # We change the value of a hyperparameter for another value
        h = np.random.choice(list(self.ev_hypers.keys()))  # We select the hyperparameter to be mutated
        # We choose two values, just in case the first one is the one already selected
        new_value = np.random.choice(self.ev_hypers[h], size=2, replace=False)
        if self.individual.descriptor_list["hypers"][h] == new_value[0]:
            self.individual.descriptor_list["hypers"][h] = new_value[1]
        else:
            self.individual.descriptor_list["hypers"][h] = new_value[0]
           
            
class MLP_Mutation(Mutation):
    
    def __init__(self, ev_hypers, max_lay, batch_normalization, drop, individual):
        super().__init__(ev_hypers, max_lay, batch_normalization, drop, individual)
        
    def mut_add_layer(self, network):# We add one layer
        layer_pos = np.random.randint(network.number_hidden_layers)+1
        lay_dims = np.random.randint(self.max_lay)+1
        init_w_function = initializations[np.random.randint(len(initializations))]
        init_a_function = activations[np.random.randint(len(activations))]
        
        if not self.drop:
            dropout = np.random.randint(0, 2)
            drop_prob = np.random.rand()
        else:
            dropout = 0
            drop_prob = 0
        
        if not self.batch_normalization:
            batch_norm = np.random.randint(0, 2)
        else:
            batch_norm = 0
    
        network.add_layer(layer_pos, lay_dims, init_w_function, init_a_function, dropout, drop_prob, batch_norm)
        
    
    def mut_del_layer(self, network): # We remove one layer
        network.remove_random_layer()

    def mut_dimension(self, network):              # We change the number of neurons in layer
        layer_pos = np.random.randint(0, self.number_hidden_layers)
        new_dim = np.random.randint(0, self.max_lay)
        
        network.change_layer_dimension(layer_pos, new_dim)
     
        
class CNN_Mutation(Mutation):

    def __init__(self, ev_hypers, max_lay, batch_normalization, drop, individual):
        super().__init__(ev_hypers, max_lay, batch_normalization, drop, individual)

    def mut_add_conv_layer(self, network):
        network.add_layer(np.random.randint(0, network.number_hidden_layers), np.random.randint(0, 3), 
                          [np.random.randint(1, network.max_stride), np.random.randint(2, network.max_filter),
                           np.random.choice(activations[1:]),  np.random.choice(initializations[1:])])
        
    def mut_del_conv_layer(self, network):
        if network.number_hidden_layers > 1:
            network.remove_random_layer()
    
    def mut_stride_conv(self, network):
        layer = np.random.randint(0, network.number_hidden_layers)
            
        new_stride = np.random.randint(1, network.max_stride)
        network.change_stride(layer, new_stride)
        
    def mut_filter_conv(self, network):
        layer = np.random.randint(0, network.number_hidden_layers)
        
        channels = np.random.randint(0, network.MAX_NUM_FILTER)
        new_filter_size = np.random.randint(2, network.max_filter)
        network.change_filters(layer, new_filter_size, channels)
            

class TCNN_Mutation(Mutation):

    def __init__(self, ev_hypers, max_lay, batch_normalization, drop, individual):
        super().__init__(ev_hypers, max_lay, batch_normalization, drop, individual)
         
    def mut_add_deconv_layer(self, network):
        network.add_layer(np.random.randint(0, network.number_hidden_layers),
                          [np.random.randint(1, network.max_stride), np.random.randint(2, network.max_filter),
                           np.random.choice(activations[1:]),  np.random.choice(initializations[1:])])
    
    def mut_del_deconv_layer(self, network):
        if network.number_hidden_layers > 1:
            network.remove_random_layer()
    
    def mut_stride_deconv(self, network):
        layer = np.random.randint(0, network.number_hidden_layers)
        new_stride = np.random.randint(1, network.max_stride)
        network.change_stride(layer, new_stride)
    
    def mut_filter_deconv(self, network):
        layer = np.random.randint(0, network.number_hidden_layers)
        
        channels = np.random.randint(0, network.MAX_NUM_FILTER)
        new_filter_size = np.random.randint(2, network.max_filter)
        network.change_filters(layer, new_filter_size, channels)

class RNN_Mutation(Mutation):
    
    def __init__(self, ev_hypers, max_lay, batch_normalization, drop, individual):
        super().__init__(ev_hypers, max_lay, batch_normalization, drop, individual)
        
    def mut_add_rnn_layer(self, network):
        layer_pos = np.random.randint(network.number_hidden_layers)
        rnn_type = np.random.choice([SimpleRNN, LSTM, GRU])
        units_in_layer = np.random.randint(1, network.max_units)
        bidirectional = np.random.choice([True, False])
        act_function = np.random.choice(activations[1:])
        init_function = np.random.choice(initializations[1:])
        network.add_layer(layer_pos, [rnn_type, units_in_layer, bidirectional, act_function, init_function])
        
    def mut_remove_rnn_layer(self, network):
        network.remove_random_layer()

    def mut_change_layer_type(self, network):
        layer_pos = np.random.randint(network.number_hidden_layers)
        network.change_layer_type(layer_pos)
        
    def mut_change_units(self, network):
        layer_pos = np.random.randint(network.number_hidden_layers)
        new_units = np.random.randint(1, network.max_units)
        network.change_units(layer_pos, new_units)
        
    def mut_change_bidirectional(self, network):
        layer_pos = np.random.randint(network.number_hidden_layers)
        network.change_bidirectional(layer_pos)
        
    def mut_change_max_units(self, network):
        max_units = np.random.randint(2,10) * 32
        network.change_max_units(max_units)
