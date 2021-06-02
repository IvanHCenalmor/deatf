import numpy as np
from evoflow.network import initializations, activations
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU

class Mutation:
    """
    This class implements the possible mutations that can be applied for a generic
    networks descriptors. This class contains the general mutations that could be 
    applied to any descriptor, subclases will have specific mutations for their own
    network.
    
    :param ev_hypers: List with the hyperparameters to be evolved.
    :param max_num_neurons: Maximum number of neurons in each layer that the descriptor can have.
    :param network: Network descriptor that will be mutated.
    :param hyperparameters: Hyperparameters that are in the mutated network.
    :param custom_mutation_list: Optinal list with the mutations that can be applied
                                 to the network. Is defined by the user with the mutations
                                 wanted, if is not defined, all possible mutations could be applied.
    """
    def __init__(self, ev_hypers, max_num_neurons, network, hyperparameters, custom_mutation_list=[]):
        self.ev_hypers = ev_hypers
        self.max_num_neurons = max_num_neurons
        self.network = network
        self.hyperparameters = hyperparameters
        if not custom_mutation_list:
            custom_mutation_list = self.methods()
        self.custom_mutation_list = custom_mutation_list

    def apply_random_mutation(self):
        """
        It selects all the available mutations for the network and after selecting
        one randomly, it applies that mutation.
        """
        type_mutation = np.random.choice(self.custom_mutation_list)
        
        self.apply_mutation(type_mutation)
        
    def apply_mutation(self, mutation):
        """
        Given a mutation method and a network, it applies that mutation to the 
        network if it is possible; if not it raises a ValueError. This function
        is mainly though for the case where the user define the mutation list, in
        order to ensure that all the described mutation can be done.
        
        :param mutation: Mutation that will be applied to the network.
        """
        if mutation in self.methods():
            eval('self.' + mutation + '()')
        else:
            raise ValueError('The mutation {} is not defined'.format(mutation))
        
    def methods(self):
        """
        Finds and returns all the possible mutations for that mutation class.
        
        :return: List with all the possible mutations that can be applied.
        """
        method_list = [method for method in dir(self.__class__) if not method.startswith('__')]
        
        method_list.remove('methods')
        method_list.remove('apply_mutation')
        method_list.remove('apply_random_mutation')
        
        
        if len(self.ev_hypers) == 0:
            method_list.remove('mut_hyper')
        if self.network.batch_norm:
            method_list.remove('mut_batch_norm')
        if self.network.dropout:
            method_list.remove('mut_dropout')
        
        return method_list
       
    def mut_weight_init(self):
        """
        Changes the weight initialization in a random layer for a new random 
        weight initialization function. 
        """
        layer_pos = np.random.randint(self.network.number_hidden_layers)
        init_w_function = initializations[np.random.randint(len(initializations))]
        self.network.change_weight_init(layer_pos, init_w_function)


    def mut_activation(self):
        """
        Changes the activation function in a random layer for a new random 
        activation function. 
        """
        layer_pos = np.random.randint(self.network.number_hidden_layers)
        init_a_function = activations[np.random.randint(len(activations))]
        self.network.change_activation(layer_pos, init_a_function)
        
    def mut_dropout(self):
        """
        Changes the dropout boolean.
        """
        self.network.change_dropout()
        
    def mut_dropout_prob(self):
        """
        Changes the dropout probabilites.
        """
        self.network.change_dropout_prob()

    def mut_batch_norm(self):
        """
        Changes the batch normalization boolean.
        """
        self.network.change_batch_norm() 
        
    def mut_hyper(self):
        """
        Changes one of the hyperparameters of the network for a new one selected
        randomly from the list of evolvable hyperparameters described in the initialization.
        """
        # We change the value of a hyperparameter for another value
        h = np.random.choice(list(self.ev_hypers.keys()))  # We select the hyperparameter to be mutated
        # We choose two values, just in case the first one is the one already selected
        new_value = np.random.choice(self.ev_hypers[h], size=2, replace=False)
        if self.hyperparameters[h] == new_value[0]:
            self.hyperparameters[h] = new_value[1]
        else:
            self.hyperparameters[h] = new_value[0]
           
            
class MLP_Mutation(Mutation):
    """
    This class implements the possible mutations that can be applied 
    specialy to the MLP descriptors. 
    
    :param ev_hypers: List with the hyperparameters to be evolved.
    :param max_num_neurons: Maximum number of neurons in each layer that the descriptor can have.
    :param network: MLP descriptor that will be mutated.
    :param hyperparameters: Hyperparameters that are in the mutated network.
    :param custom_mutation_list: Optinal list with the mutations that can be applied
                                 to the network. Is defined by the user with the mutations
                                 wanted, if is not defined, all possible mutations could be applied.
    """
    def __init__(self, ev_hypers, max_num_neurons, network, hyperparameters, custom_mutation_list=[]):
        super().__init__(ev_hypers, max_num_neurons, network, hyperparameters, custom_mutation_list)
        
    def mut_add_layer(self):
        """
        A new dense layer is added to the MLP descriptor.
        """
        layer_pos = np.random.randint(self.network.number_hidden_layers)+1
        lay_dims = np.random.randint(self.max_num_neurons)+1
        init_function = initializations[np.random.randint(len(initializations))]
        act_function = activations[np.random.randint(len(activations))]
        drop_prob = np.random.rand()
    
        self.network.add_layer(layer_pos, lay_dims, init_function, act_function, drop_prob)
        
    
    def mut_remove_layer(self): 
        """
        Removes a random layer from the MLP descriptor.
        """
        self.network.remove_random_layer()

    def mut_dimension(self):
        """
        Number of neurons of a random layer are changed to a new random
        number of neurons from 1 to maximum number of neurons described 
        in the initialization.
        """
        layer_pos = np.random.randint(0, self.network.number_hidden_layers)
        new_dim = np.random.randint(1, self.max_num_neurons)
        
        self.network.change_layer_dimension(layer_pos, new_dim)
     
        
class CNN_Mutation(Mutation):
    """
    This class implements the possible mutations that can be applied 
    specialy to the CNN descriptors. 
    
    :param ev_hypers: List with the hyperparameters to be evolved.
    :param max_num_neurons: Maximum number of neurons in each layer that the descriptor can have.
    :param network: CNN descriptor that will be mutated.
    :param hyperparameters: Hyperparameters that are in the mutated network.
    :param custom_mutation_list: Optinal list with the mutations that can be applied
                                 to the network. Is defined by the user with the mutations
                                 wanted, if is not defined, all possible mutations could be applied.
    """
    def __init__(self, ev_hypers, max_num_neurons, network, hyperparameters, custom_mutation_list=[]):
        super().__init__(ev_hypers, max_num_neurons, network, hyperparameters, custom_mutation_list)

    def mut_add_conv_layer(self):
        """
        A new convolutional or pooling layer is added to the CNN descriptor.
        """
        new_layer_pos = np.random.randint(0, self.network.number_hidden_layers)
        new_lay_type = np.random.randint(0, 3)
        
        new_filters = np.random.randint(1, self.network.max_stride)
        new_strides = np.random.randint(2, self.network.max_filter)
        new_act_functions = np.random.choice(activations[1:])
        new_init_functions = np.random.choice(initializations[1:])
        lay_params = [new_filters, new_strides, new_act_functions, new_init_functions]
        
        self.network.add_layer(new_layer_pos, new_lay_type, lay_params)
        
    def mut_remove_conv_layer(self):
        """
        Removes a random layer from the CNN descriptor if it is possible (if there is only
        one layer, it can not be removed).        
        """
        if self.network.number_hidden_layers > 1:
            self.network.remove_random_layer()
    
    def mut_stride_conv(self):
        """
        Stride of a random layer is changed to a new random
        stride from 1 to maximum stride size described 
        in the network descriptor.
        """
        layer_pos = np.random.randint(0, self.network.number_hidden_layers)
            
        new_stride = np.random.randint(1, self.network.max_stride)
        self.network.change_stride(layer_pos, new_stride)
        
    def mut_filter_conv(self):
        """
        Filter of a random layer is changed to a new random
        filter with size from 2 to maximum filter size described 
        in the network descriptor and with channels from 1 to 65.
        """
        layer_pos = np.random.randint(0, self.network.number_hidden_layers)
        
        new_filter_size = np.random.randint(2, self.network.max_filter)
        channels = np.random.randint(1, self.network.MAX_NUM_FILTER)
        self.network.change_filters(layer_pos, new_filter_size, channels)
            

class TCNN_Mutation(Mutation):
    """
    This class implements the possible mutations that can be applied 
    specialy to the TCNN descriptors. 
    
    :param ev_hypers: List with the hyperparameters to be evolved.
    :param max_num_neurons: Maximum number of neurons in each layer that the descriptor can have.
    :param network: TCNN descriptor that will be mutated.
    :param hyperparameters: Hyperparameters that are in the mutated network.
    :param custom_mutation_list: Optinal list with the mutations that can be applied
                                 to the network. Is defined by the user with the mutations
                                 wanted, if is not defined, all possible mutations could be applied.
    """
    def __init__(self, ev_hypers, max_num_neurons, network, hyperparameters, custom_mutation_list=[]):
        super().__init__(ev_hypers, max_num_neurons, network, hyperparameters, custom_mutation_list)
         
    def mut_add_deconv_layer(self):
        """
        A new transposed convolutional layer is added to the TCNN descriptor.
        """
        new_layer_pos = np.random.randint(0, self.network.number_hidden_layers)
        
        new_filters = np.random.randint(1, self.network.max_stride)
        new_strides = np.random.randint(2, self.network.max_filter)
        new_act_functions = np.random.choice(activations[1:])
        new_init_functions = np.random.choice(initializations[1:])
        lay_params = [new_filters, new_strides, new_act_functions, new_init_functions]
        
        self.network.add_layer(new_layer_pos, lay_params)
    
    def mut_remove_deconv_layer(self):
        """
        Removes a random layer from the TCNN descriptor if it is possible (if there is only
        one layer, it can not be removed).        
        """
        if self.network.number_hidden_layers > 1:
            self.network.remove_random_layer()
    
    def mut_stride_deconv(self):
        """
        Stride of a random layer is changed to a new random
        stride from 1 to maximum stride size described 
        in the network descriptor.
        """
        layer_pos = np.random.randint(0, self.network.number_hidden_layers)
        new_stride = np.random.randint(1, self.network.max_stride)
        self.network.change_stride(layer_pos, new_stride)
    
    def mut_filter_deconv(self):
        """
        Filter of a random layer is changed to a new random
        filter with size from 2 to maximum filter size described 
        in the network descriptor and with channels from 1 to 65.
        """
        layer_pos = np.random.randint(0, self.network.number_hidden_layers)
        
        channels = np.random.randint(0, self.network.MAX_NUM_FILTER)
        new_filter_size = np.random.randint(2, self.network.max_filter)
        self.network.change_filters(layer_pos, new_filter_size, channels)

class RNN_Mutation(Mutation):
    """
    This class implements the possible mutations that can be applied 
    specialy to the RNN descriptors. 
    
    :param ev_hypers: List with the hyperparameters to be evolved.
    :param max_num_neurons: Maximum number of neurons in each layer that the descriptor can have.
    :param network: RNN descriptor that will be mutated.
    :param hyperparameters: Hyperparameters that are in the mutated network.
    :param custom_mutation_list: Optinal list with the mutations that can be applied
                                 to the network. Is defined by the user with the mutations
                                 wanted, if is not defined, all possible mutations could be applied.
    """
    def __init__(self, ev_hypers, max_num_neurons, network, hyperparameters, custom_mutation_list=[]):
        super().__init__(ev_hypers, max_num_neurons, network, hyperparameters, custom_mutation_list)
        
    def mut_add_rnn_layer(self):
        """
        A new recurrent layer is added to the RNN descriptor.        
        """
        layer_pos = np.random.randint(self.network.number_hidden_layers)
        
        rnn_type = np.random.choice([SimpleRNN, LSTM, GRU])
        units_in_layer = np.random.randint(1, self.network.max_units)
        bidirectional = np.random.choice([True, False])
        act_function = np.random.choice(activations[1:])
        init_function = np.random.choice(initializations[1:])
        
        self.network.add_layer(layer_pos, [rnn_type, units_in_layer, bidirectional, act_function, init_function])
        
    def mut_remove_rnn_layer(self):
        """
        Removes a random layer from the RNN descriptor.   
        """
        self.network.remove_random_layer()

    def mut_change_layer_type(self):
        """
        Type of a random layer is changed to a new random type.
        """
        layer_pos = np.random.randint(self.network.number_hidden_layers)
        self.network.change_layer_type(layer_pos)
        
    def mut_change_units(self):
        """
        Number of units in a random layer is changed to a new random 
        number of units from 1 to maximum number of units described in 
        network descriptor.        
        """
        layer_pos = np.random.randint(self.network.number_hidden_layers)
        new_units = np.random.randint(1, self.network.max_units)
        self.network.change_units(layer_pos, new_units)
        
    def mut_change_bidirectional(self):
        """
        A random layer is changed to be biderctional if is not; otherwise,
        it stops beeing bidirectional.        
        """
        layer_pos = np.random.randint(self.network.number_hidden_layers)
        self.network.change_bidirectional(layer_pos)
        
    def mut_change_max_units(self):
        """
        Maximum number of units allowed in the network is change by a new
        randomly selected one from a value between 2 and 10 multiplied by 32.        
        """
        max_units = np.random.randint(2,10) * 32
        self.network.change_max_units(max_units)
