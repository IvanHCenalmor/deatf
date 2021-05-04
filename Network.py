import tensorflow as tf
import numpy as np
from functools import reduce
import os
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, \
                                    MaxPooling2D, AveragePooling2D, Bidirectional, SimpleRNN, LSTM, GRU
from tensorflow.keras.initializers import RandomNormal, RandomUniform, GlorotNormal, GlorotUniform

activations = [None, tf.nn.relu, tf.nn.elu, tf.nn.softplus, tf.nn.softsign, tf.sigmoid, tf.nn.tanh]
initializations = [RandomNormal, RandomUniform, GlorotNormal, GlorotUniform]

class NetworkDescriptor:

    def __init__(self, number_hidden_layers=1, input_dim=1, output_dim=1, init_functions=None, act_functions=None, 
                 dropout=False, dropout_probs=(), batch_norm=False):
        """
        This class implements the descriptor of a generic network. Subclasses of this are the ones evolved.
        :param number_hidden_layers: Number of hidden layers in the network
        :param input_dim: Dimension of the input data (can have one or more dimensions)
        :param output_dim: Expected output of the network (similarly, can have one or more dimensions)
        :param init_functions: Weight initialization functions. Can have different ranges, depending on the subclass
        :param act_functions: Activation functions to be applied after each layer
        :param dropout: A 0-1 array of the length number_hidden_layers indicating  whether a dropout "layer" is to be
        applied AFTER the activation function
        :param batch_norm: A 0-1 array of the length number_hidden_layers indicating  whether a batch normalization
        "layer" is to be applied BEFORE the activation function
        """
        self.number_hidden_layers = number_hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init_functions = init_functions
        self.act_functions = act_functions
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.dropout_probs = dropout_probs

    def remove_layer(self, _):  # Defined just in case the user redefines classes and forgets to define this function
        pass

    def remove_random_layer(self):
        layer_pos = np.random.randint(self.number_hidden_layers)
        self.remove_layer(layer_pos)

    def change_activation(self, layer_pos, new_act_fn):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_hidden_layers:
            return
        self.act_functions[layer_pos] = new_act_fn

    def change_weight_init(self, layer_pos, new_weight_fn):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_hidden_layers:
            return
        self.init_functions[layer_pos] = new_weight_fn

    def change_dropout(self):
        # Change dropout
        self.dropout = not self.dropout
    
    def change_dropout_prob(self):
        self.dropout_probs = np.random.rand(self.number_hidden_layers+1)
    
    def change_batch_norm(self):
        # Change batch normalization
        self.batch_norm = not self.batch_norm

class MLPDescriptor(NetworkDescriptor):
    def __init__(self, number_hidden_layers=1, input_dim=1, output_dim=1,  dims=None, init_functions=None, 
                 act_functions=None, dropout=False, batch_norm=False):
        """
        :param number_hidden_layers: Number of hidden layers in the network
        :param input_dim: Dimension of the input data (can have one or more dimensions)
        :param output_dim: Expected output of the network (similarly, can have one or more dimensions)
        :param init_functions: Weight initialization functions. Can have different ranges, depending on the subclass
        :param dims: Number of neurons in each layer
        :param act_functions: Activation functions to be applied after each layer
        :param dropout: A 0-1 array of the length number_hidden_layers indicating  whether a dropout "layer" is to be
        applied AFTER the activation function
        :param batch_norm: A 0-1 array of the length number_hidden_layers indicating  whether a batch normalization
        "layer" is to be applied BEFORE the activation function
        """

        super().__init__(number_hidden_layers=number_hidden_layers, input_dim=input_dim, output_dim=output_dim, init_functions=init_functions, act_functions=act_functions, dropout=dropout, batch_norm=batch_norm)
        self.dims = dims  # Number of neurons in each layer

    def random_init(self, input_size=None, output_size=None, nlayers=None, max_layer_size=None, 
                    _=None, __=None, dropout=None, batch_norm=None):

        # If the incoming/outgoing sizes have more than one dimension compute the size of the flattened sizes
        if input_size is not None:
            if hasattr(input_size, '__iter__'):
                self.input_dim = reduce(lambda x, y: x*y, input_size)
            else:
                self.input_dim = input_size
        if output_size is not None:
            if hasattr(output_size, '__iter__'):
                self.output_dim = reduce(lambda x, y: x*y, output_size)
            else:
                self.output_dim = output_size

        # Random initialization
        if nlayers is not None and max_layer_size is not None:
            self.number_hidden_layers = np.random.randint(nlayers)+1
            self.dims = [np.random.randint(4, max_layer_size)+1 for _ in range(self.number_hidden_layers)]
            self.init_functions = np.random.choice(initializations, size=self.number_hidden_layers+1)
            self.act_functions = np.random.choice(activations, size=self.number_hidden_layers+1)
        
        if batch_norm is not None and batch_norm:
            self.batch_norm = np.random.choice([True, False])
            
        if dropout is not None and dropout:
            self.dropout = np.random.choice([True, False])
            self.dropout_probs = np.random.rand(self.number_hidden_layers+1)
        else:
            self.dropout_probs = np.zeros(self.number_hidden_layers+1)

    def add_layer(self, layer_pos, lay_dims, init_w_function, init_a_function, dropout, drop_prob, batch_norm):
        """
        This function adds a layer in the layer_pos position
        :param layer_pos: Position of the layer
        :param lay_dims: Number of neurons in the layer
        :param init_w_function: function for initializing the layer
        :param init_a_function: activation function to be applied after the layer
        :param dropout: Whether dropout is applied or not in the layer
        :param drop_prob: probability of dropout
        :param batch_norm: Whether batch normalization is applied after the layer
        :return:
        """

        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos >= self.number_hidden_layers:
            return

        # We create the new layer and add it to the network descriptor
        self.dims = np.insert(self.dims, layer_pos, lay_dims)
        self.init_functions = np.insert(self.init_functions, layer_pos, init_w_function)
        self.act_functions = np.insert(self.act_functions, layer_pos, init_a_function)
        self.number_hidden_layers = self.number_hidden_layers + 1
        self.dropout_probs = np.insert(self.dropout_probs, layer_pos, drop_prob)

    def remove_layer(self, layer_pos):
        """
        This function deletes a layer
        :param layer_pos: Position of the layer to be deleted
        :return:
        """

        # If not within feasible bounds, return
        if layer_pos <= 1 or layer_pos > self.number_hidden_layers:
            return

        # We set the number of input and output dimensions for the layer to be
        # added and for the ones in the architecture that will be connected to it

        # We delete the layer in pos layer_pos
        self.dims = np.delete(self.dims, layer_pos)
        self.init_functions = np.delete(self.init_functions, layer_pos)
        self.act_functions = np.delete(self.act_functions, layer_pos)
        self.dropout_probs = np.delete(self.dropout_probs, layer_pos)

        # Finally the number of hidden layers is updated
        self.number_hidden_layers = self.number_hidden_layers - 1

    def change_layer_dimension(self, new_dim):

        layer_pos = np.random.randint(0, self.number_hidden_layers)
        self.dims[layer_pos] = new_dim

    def print_components(self, identifier):
        print(identifier, ' n_hid:', self.number_hidden_layers)
        print(identifier, ' Dims:', self.dims)
        print(identifier, ' Init:', self.init_functions)
        print(identifier, ' Act:', self.act_functions)

    def codify_components(self, max_hidden_layers, ref_list_init_functions, ref_list_act_functions):

        max_total_layers = max_hidden_layers + 1
        # The first two elements of code are the number of layers and number of loops
        code = [self.number_hidden_layers]

        # We add all the layer dimension and fill with zeros all positions until max_layers
        code = code + self.dims + [-1]*(max_total_layers-len(self.dims))

        # We add the indices of init_functions in each layer
        # and fill with zeros all positions until max_layers
        aux_f = []
        for init_f in self.init_functions:
            aux_f.append(ref_list_init_functions.index(init_f))
        code = code + aux_f + [-1]*(max_total_layers-len(aux_f))

        # We add the indices of act_functions in each layer
        # and fill with zeros all positions until max_layers
        aux_a = []
        for act_f in self.act_functions:
            aux_a.append(ref_list_act_functions.index(act_f))
        code = code + aux_a + [-1]*(max_total_layers-len(aux_a))

        return code


class ConvDescriptor(NetworkDescriptor):

    MAX_NUM_FILTER = 65

    def __init__(self, number_hidden_layers=2, input_dim=(28, 28, 3), output_dim=(7, 7, 1), op_type=(2, 1), 
                 max_filter=3, max_stride=2, filters=((3, 3, 2), (3, 3, 2)), strides=((1, 1, 1), (1, 1, 1)), 
                 list_init_functions=(0, 0), list_act_functions=(0, 0), dropout=(), batch_norm=()):
        """
        Descriptor for convolutional cells
        :param number_hidden_layers: Number of hidden layers (it's changed afterwards)
        :param input_dim: Dimension of the input
        :param output_dim: Expected dimension of the output (could be greater)
        :param op_type: Type of layer (Mean pooling, max pooling, or convolutional. it's changed afterwards)
        :param filters: list of dimensions of filters (it's changed afterwards)
        :param strides: list of strides (it's changed afterwards)
        :param list_init_functions: list of initialization functions of the filter weights (it's changed afterwards)
        :param list_act_functions: list of activation functions after filters (it's changed afterwards)
        :param dropout: list of booleans defining whether dropout is applied to each layer (it's changed afterwards)
        :param batch_norm: list of booleans defining whether batch normalization is applied to each layer (it's changed afterwards)
        """

        super().__init__(number_hidden_layers=number_hidden_layers, input_dim=input_dim, output_dim=output_dim, 
                         init_functions=list_init_functions, act_functions=list_act_functions, dropout=False, 
                         batch_norm=False)
        self.layers = op_type
        self.filters = filters
        self.strides = strides
        self.max_stride = max_stride
        self.max_filter = max_filter
        
    def random_init(self, input_size, output_size, nlayers, _, max_stride, max_filter, dropout, batch_norm):
        """
        This function randomly initializes the descriptor. This function is susceptible of being modified by the user with specific creation needs
        :param input_size:  Dimension of the input
        :param output_size: Expected dimension of the output (could be greater)
        :param nlayers: maximum number of layers
        :param _: unused
        :param max_stride: maximum stride possible (used as 2)
        :param max_filter: maximum filter size possible (used as 3)
        :param dropout: Whether dropout is a possibility in the network
        :param batch_norm: Whether batch normalization is a possibility in the network
        :return:
        """

        self.input_dim = input_size
        self.output_dim = output_size

        self.number_hidden_layers = np.random.randint(nlayers)+1
        self.layers = []
        self.max_stride = max_stride
        self.strides = []
        self.max_filter = max_filter
        self.filters = []
        self.init_functions = []
        self.act_functions = []
        
        i = 0
        while i < nlayers:
            new_layer = np.random.choice([0, 1, 2])
            
            self.layers += [2]
            self.filters += [np.array([np.random.randint(2, max_filter)] * 2 + [np.random.randint(3, self.MAX_NUM_FILTER)])]
            self.strides += [np.array([np.random.randint(1, max_stride)] * 2 + [1])]
            shape = calculate_CNN_shape(self.input_dim, self.filters, self.strides, -1)
            if shape[0] < 2 or shape[1] < 2 or np.prod(shape) < self.output_dim:  # If the blob size is too small
                self.number_hidden_layers = i
                self.layers = self.layers[:-1]
                self.filters = self.filters[:-1]
                self.strides = self.strides[:-1]
                break
            
            self.init_functions += [np.random.choice(initializations[1:])]
            self.act_functions += [np.random.choice(activations)]
            
            if new_layer != 2:
                self.layers += [new_layer]
                self.filters += [np.array([np.random.randint(2, max_filter)] * 2 + [np.random.randint(3, self.MAX_NUM_FILTER)])]
                self.strides += [np.array([np.random.randint(1, max_stride)] * 2 + [1])]
                shape = calculate_CNN_shape(self.input_dim, self.filters, self.strides, -1)
                if shape[0] < 2 or shape[1] < 2 or np.prod(shape) < self.output_dim:  # If the blob size is too small
                    self.number_hidden_layers = i
                    self.layers = self.layers[:-2]
                    self.filters = self.filters[:-2]
                    self.strides = self.strides[:-2]
                    self.init_functions = self.init_functions[:-1]
                    self.act_functions = self.act_functions[:-1]
                    break
                i += 1
                self.init_functions += [np.random.choice(initializations[1:])]
                self.act_functions += [np.random.choice(activations)]
                
            i += 1
            
        if batch_norm is not None and not batch_norm:
            self.batch_norm = np.random.choice([True, False])

    def add_layer(self, layer_pos, lay_type, lay_params):
        """
        This function adds a layer in the layer_pos position
        :param layer_pos: Position of the layer
        :param lay_type: Type of operation (0, 1: pooling, 2 convolutional)
        :param lay_params: sizes of the *filters*.
        :return:
        """
        
        aux_filters = copy.deepcopy(self.filters)
        aux_strides = copy.deepcopy(self.strides)
        aux_number_hidden_layers = self.number_hidden_layers
        
        if self.layers[layer_pos] != 2:
            layer_pos += 1
        
        aux_number_hidden_layers += 1
        aux_filters.insert(layer_pos, np.array([lay_params[1], lay_params[1], np.random.randint(0, self.MAX_NUM_FILTER)]))
        aux_strides.insert(layer_pos, np.array([lay_params[0], lay_params[0], 1]))
        if lay_type < 2:
            aux_number_hidden_layers += 1
            pool_params = [1, np.random.randint(2, 4), np.random.choice(activations[1:]), np.random.choice(initializations[1:])]
            
            aux_filters.insert(layer_pos, np.array([pool_params[1], pool_params[1], np.random.randint(0, self.MAX_NUM_FILTER)]))
            aux_strides.insert(layer_pos, np.array([pool_params[0], pool_params[0], 1]))
            
        if calculate_CNN_shape(self.input_dim, aux_filters, aux_strides, aux_number_hidden_layers)[0] > 0:
        
            self.layers.insert(layer_pos, 2)
            # A convolutional layer is added always
            self.act_functions.insert(layer_pos, lay_params[2])
            self.init_functions.insert(layer_pos, lay_params[3])
    
            if lay_type < 2:
                # If a max_pool or avg_pool has to be added
                self.layers.insert(layer_pos+1, lay_type)
                
                self.act_functions.insert(layer_pos+1, None)
                self.init_functions.insert(layer_pos+1, None)               
                
            self.filters = aux_filters
            self.strides = aux_strides
            self.number_hidden_layers = aux_number_hidden_layers           

    def remove_layer(self, layer_pos):
        """
        This function deletes a layer
        :param layer_pos: Position of the layer to be deleted
        :return:
        """
 
        self.layers.pop(layer_pos)
        self.act_functions.pop(layer_pos)
        self.init_functions.pop(layer_pos)
        self.filters.pop(layer_pos)
        self.strides.pop(layer_pos)
        self.number_hidden_layers -= 1
        
        if layer_pos < self.number_hidden_layers:
            if self.layers[layer_pos] != 2:
                self.layers.pop(layer_pos)
                self.act_functions.pop(layer_pos)
                self.init_functions.pop(layer_pos)
                self.filters.pop(layer_pos)
                self.strides.pop(layer_pos)
                self.number_hidden_layers -= 1

    def remove_random_layer(self):
        """
        Select a random layer and execute the deletion
        :return:
        """
        if self.number_hidden_layers > 1:
            layer_pos = np.random.randint(self.number_hidden_layers)
            self.remove_layer(layer_pos)
            return 0
        else:
            return -1

    def change_filters(self, layer_pos, new_kernel_size, new_channel):
        """
        Change the size of one layer filter
        :param layer_pos: Position of the filter to be changed
        :param new_kernel_size: Height and width of the filter (only square filters are allowed)
        :param new_channel: Number of output channels
        :return:
        """
        aux_filters = copy.deepcopy(self.filters)
        aux_filters[layer_pos][0] = new_kernel_size
        aux_filters[layer_pos][1] = new_kernel_size
        aux_filters[layer_pos][2] = new_channel
        
        if calculate_CNN_shape(self.input_dim, aux_filters, self.strides, self.number_hidden_layers)[0] > 0:
            self.filters = aux_filters

    def change_stride(self, layer_pos, new_stride):
        """
        Change the stride of a filter in a layer
        :param layer_pos: Layer which stride is changed
        :param new_stride: self-explanatory
        :return:
        """
        aux_strides = copy.deepcopy(self.strides)
        aux_strides[layer_pos][0] = new_stride
        aux_strides[layer_pos][1] = new_stride
        
        if calculate_CNN_shape(self.input_dim, self.filters, aux_strides, self.number_hidden_layers)[0] > 0:
            self.strides = aux_strides
            
    def print_components(self, identifier):
        print(identifier, ' n_conv:', len([x for x in self.filters if not x == -1]))
        print(identifier, ' n_pool:', len([x for x in self.filters if x == -1]))
        print(identifier, ' Init:', self.init_functions)
        print(identifier, ' Act:', self.act_functions)
        print(identifier, ' filters:', self.filters)
        print(identifier, ' strides:', self.strides)

    def codify_components(self):

        filters = [str(x) for x in self.filters]
        init_funcs = [str(x) for x in self.init_functions]
        act_funcs = [str(x) for x in self.act_functions]
        sizes = [[str(y) for y in x] for x in self.filters]
        strides = [str(x) for x in self.strides]
        return str(self.input_dim) + "_" + str(self.output_dim) + "_" + ",".join(filters) + "*" + \
                    ",".join(["/".join(szs) for szs in sizes]) + "*" + ",".join(strides) + "_" + \
                    ",".join(init_funcs) + "_" + ",".join(act_funcs)


class TConvDescriptor(NetworkDescriptor):
    
    MAX_NUM_FILTER = 65
    
    def __init__(self, number_hidden_layers=2, input_dim=(7, 7, 50), output_dim=(28, 28, 3), 
                 max_filter=3, filters=((3, 3, 2), (3, 3, 2)), max_stride=2, strides=((1, 1, 1), (1, 1, 1)), 
                 list_init_functions=(0, 0), list_act_functions=(0, 0), dropout=(), batch_norm=()):
        """
       Descriptor for transposed convolutional cells
       :param number_hidden_layers: Number of hidden layers (it's changed afterwards)
       :param input_dim: Dimension of the input
       :param output_dim: Expected dimension of the output (could be greater)
       :param filters: list of dimensions of filters (it's changed afterwards)
       :param strides: list of strides (it's changed afterwards)
       :param list_init_functions: list of initialization functions of the filter weights (it's changed afterwards)
       :param list_act_functions: list of activation functions after filters (it's changed afterwards)
       :param dropout: list of booleans defining whether dropout is applied to each layer (it's changed afterwards)
       :param batch_norm: list of booleans defining whether batch normalization is applied to each layer (it's changed afterwards)
       """

        super().__init__(number_hidden_layers=number_hidden_layers, input_dim=input_dim, output_dim=output_dim, 
                         init_functions=list_init_functions, act_functions=list_act_functions, dropout=False, 
                         batch_norm=False)

        self.max_stride = max_stride
        self.strides = strides
        self.max_filter = max_filter
        self.filters = filters

    def random_init(self, input_size, output_size, _, __, max_stride, max_filter, dropout, batch_norm):
        """
        This function randomly initializes the descriptor. This function is susceptible of being modified by the user with specific creation needs
        :param input_size:  Dimension of the input
        :param output_size: Expected dimension of the output (could be greater)
        :param _: unused
        :param __: unused
        :param max_stride: maximum stride possible (used as 2)
        :param max_filter: maximum filter size possible (used as 3)
        :param dropout: Whether dropout is a possibility in the network
        :param batch_norm: Whether batch normalization is a possibility in the network
        :return:
        """
        self.input_dim = input_size
        self.output_dim = output_size

        # Random initialization
        
        self.max_stride = max_stride
        self.strides = []
        self.max_filter = max_filter
        self.filters = []
        self.init_functions = []
        self.act_functions = []
        for i in range(300):
            self.strides += [np.array([np.random.randint(1, max_stride)] * 2 + [1])]
            self.filters += [np.array([np.random.randint(2, max_filter)] * 2 + [np.random.randint(3, self.MAX_NUM_FILTER)])]
            self.init_functions += [np.random.choice(initializations[1:])]
            self.act_functions += [np.random.choice(activations)]

            shape = calculate_TCNN_shape(self.input_dim, self.filters, self.strides, -1)
            
            # Once the expected shape is exceeded, we have enough layers
            if shape[0] >= self.output_dim[0] and shape[1] >= self.output_dim[1]:  
                prev_shape = calculate_TCNN_shape(self.input_dim, self.filters[:-1], self.strides[:-1], -1)
                while prev_shape[0] * self.strides[-1][0] > self.output_dim[0]:
                    self.strides[-1] = np.array([self.strides[-1][0] - 1, self.strides[-1][1] - 1, self.strides[-1][2]])
                desired_filter_size = self.output_dim[0] - (prev_shape[0] - 1) * self.strides[-1][0]
                self.filters[-1] = np.array([desired_filter_size, desired_filter_size, self.output_dim[2]])
                self.number_hidden_layers = i+1
                break

        if batch_norm is not None and not batch_norm:
            self.batch_norm = np.random.choice([True, False])

    def add_layer(self, layer_pos, lay_params):
        """
        This function adds a layer in the layer_pos position
        :param layer_pos: Position of the layer
        :param lay_params: sizes of the filters.
        :return:
        """
        self.number_hidden_layers += 1
        self.strides.insert(layer_pos, [lay_params[0], lay_params[0], 1])
        self.filters.insert(layer_pos, [lay_params[1], lay_params[1], np.random.randint(0, self.MAX_NUM_FILTER)])
        self.act_functions.insert(layer_pos, lay_params[2])
        self.init_functions.insert(layer_pos, lay_params[3])
        
        output = calculate_TCNN_shape(self.input_dim, self.filters[:-1], self.strides[:-1], -1)
        
        while output[0] * self.strides[-1][0] > self.output_dim[0]:
            self.strides[-1] = np.array([self.strides[-1][0] - 1, self.strides[-1][1] - 1, self.strides[-1][2]])
        desired_filter_size = self.output_dim[0] - (output[0] - 1) * self.strides[-1][0]
        self.filters[-1] = np.array([desired_filter_size, desired_filter_size, self.output_dim[2]])
        
        return 0

    def remove_layer(self, layer_pos):
        """
        This function deletes a layer
        :param layer_pos: Position of the layer to be deleted
        :return:
        """

        self.filters.pop(layer_pos)
        self.act_functions.pop(layer_pos)
        self.init_functions.pop(layer_pos)
        self.strides.pop(layer_pos)
        self.number_hidden_layers -= 1
        
        output = calculate_TCNN_shape(self.input_dim, self.filters[:-1], self.strides[:-1], -1)
        
        while output[0] * self.strides[-1][0] > self.output_dim[0]:
            self.strides[-1] = np.array([self.strides[-1][0] - 1, self.strides[-1][1] - 1, self.strides[-1][2]])
        desired_filter_size = self.output_dim[0] - (output[0] - 1) * self.strides[-1][0]
        self.filters[-1] = np.array([desired_filter_size, desired_filter_size, self.output_dim[2]])

    def remove_random_layer(self):
        if self.number_hidden_layers > 1:
            layer_pos = np.random.randint(self.number_hidden_layers)
            self.remove_layer(layer_pos)
            return 0
        else:
            return -1
        
    def change_activation(self, layer_pos, new_act_fn):
        self.act_functions[layer_pos] = new_act_fn

    def change_weight_init(self, layer_pos, new_weight_fn):
        self.init_functions[layer_pos] = new_weight_fn

    def change_filters(self, layer_pos, new_kernel_size, new_channel):
        self.filters[layer_pos][0] = new_kernel_size
        self.filters[layer_pos][1] = new_kernel_size
        self.filters[layer_pos][2] = new_channel
        
        output = calculate_TCNN_shape(self.input_dim, self.filters[:-1], self.strides[:-1], -1)
        
        while output[0] * self.strides[-1][0] > self.output_dim[0]:
            self.strides[-1] = np.array([self.strides[-1][0] - 1, self.strides[-1][1] - 1, self.strides[-1][2]])
        desired_filter_size = self.output_dim[0] - (output[0] - 1) * self.strides[-1][0]
        self.filters[-1] = np.array([desired_filter_size, desired_filter_size, self.output_dim[2]])

    def change_stride(self, layer_pos, new_stride):
        
        self.strides[layer_pos][0] = new_stride
        self.strides[layer_pos][1] = new_stride
        
        output = calculate_TCNN_shape(self.input_dim, self.filters[:-1], self.strides[:-1], -1)
        
        while output[0] * self.strides[-1][0] > self.output_dim[0]:
            self.strides[-1] = np.array([self.strides[-1][0] - 1, self.strides[-1][1] - 1, self.strides[-1][2]])
        desired_filter_size = self.output_dim[0] - (output[0] - 1) * self.strides[-1][0]
        self.filters[-1] = np.array([desired_filter_size, desired_filter_size, self.output_dim[2]])

    def print_components(self, identifier):
        print(identifier, ' n_conv:', len([x for x in self.filters if not x == -1]))
        print(identifier, ' n_pool:', len([x for x in self.filters if x == -1]))
        print(identifier, ' Init:', self.init_functions)
        print(identifier, ' Act:', self.act_functions)
        print(identifier, ' filters:', self.filters)
        print(identifier, ' strides:', self.strides)

    def codify_components(self):

        filters = [str(x) for x in self.filters]
        init_funcs = [str(x) for x in self.init_functions]
        act_funcs = [str(x) for x in self.act_functions]
        sizes = [[str(y) for y in x] for x in self.filters]
        strides = [str(x) for x in self.strides]
        return str(self.input_dim) + "_" + str(self.output_dim) + "_" + ",".join(filters) + "*" + \
                    ",".join(["/".join(szs) for szs in sizes]) + "*" + ",".join(strides) + "_" + \
                    ",".join(init_funcs) + "_" + ",".join(act_funcs)

class RNNDescriptor(NetworkDescriptor):
    def __init__(self, number_hidden_layers=1, input_dim=1, output_dim=1, init_functions=None,
                 rnn_layers = [SimpleRNN], bidirectional=[False], units_in_layer=[64], max_units=128, 
                 act_functions=None, dropout=False, batch_norm=False):
        
        super().__init__(number_hidden_layers=number_hidden_layers, input_dim=input_dim, output_dim=output_dim, 
             init_functions=init_functions, act_functions=act_functions, dropout=dropout, batch_norm=batch_norm)
        self.rnn_layers = rnn_layers
        self.units_in_layer = [min(unit, max_units) for unit in units_in_layer]
        self.max_units = max_units
        self.bidirectional = bidirectional
        
    def random_init(self, input_size, output_size, nlayers, max_layer_size, _, __, dropout, batch_norm):
        
        self.input_dim = input_size
        self.output_dim = output_size
        
        self.max_units = np.random.randint(2,4) * 32
        
        # Random initialization
        self.number_hidden_layers = np.random.randint(nlayers)+1
        self.units_in_layer = [np.random.randint(1, self.max_units)+1 for _ in range(self.number_hidden_layers)]
        self.init_functions = list(np.random.choice(initializations, size=self.number_hidden_layers+1))
        self.act_functions = list(np.random.choice(activations, size=self.number_hidden_layers+1))
        
        self.rnn_layers = list(np.random.choice([SimpleRNN, LSTM, GRU], size=self.number_hidden_layers+1))
        self.bidirectional = list(np.random.choice([True, False], size=self.number_hidden_layers+1))
        
        if dropout is not None and dropout:
            self.dropout = np.random.choice([True, False])
            self.dropout_probs = np.random.rand(self.number_hidden_layers+1)
        else:
            self.dropout_probs = np.zeros(self.number_hidden_layers+1)
    
    def add_layer(self, layer_pos, lay_params):
        self.number_hidden_layers += 1
        self.rnn_layers.insert(layer_pos, lay_params[0])
        self.units_in_layer.insert(layer_pos, min(lay_params[1], self.max_units))
        self.bidirectional.insert(layer_pos, lay_params[2])
        self.act_functions.insert(layer_pos, lay_params[3])
        self.init_functions.insert(layer_pos, lay_params[4])
    
    def remove_layer(self, layer_pos):
        self.number_hidden_layers -= 1
        self.rnn_layers.pop(layer_pos)
        self.units_in_layer.pop(layer_pos)
        self.bidirectional.pop(layer_pos)
        self.act_functions.pop(layer_pos)
        self.init_functions.pop(layer_pos)
    
    def remove_random_layer(self):
        if self.number_hidden_layers > 1:
            layer_pos = np.random.randint(self.number_hidden_layers)
            self.remove_layer(layer_pos)
            return 0
        else:
            return -1
        
    def change_layer_type(self, layer_pos):
        layer_type = self.rnn_layers[layer_pos]
        possible_types = [SimpleRNN, LSTM, GRU]
        possible_types.remove(layer_type)
        self.rnn_layers[layer_pos] = np.random.choice(possible_types)
        
    def change_units(self, layer_pos, new_units):
        self.units_in_layer[layer_pos] = new_units
        
    def change_bidirectional(self, layer_pos):
        self.bidirectional[layer_pos] = not self.bidirectional[layer_pos]
        
    def change_max_units(self, max_units):
        self.max_units = max_units
        self.units_in_layer = [min(unit, self.max_units) for unit in self.units_in_layer]        
    
    def print_components(self, identifier):
        print(identifier, ' RNN_layers:', self.rnn_layers)
        print(identifier, ' Max units:', self.max_units)
        print(identifier, ' Units:', self.units_in_layer)
        print(identifier, ' Bidirectional:', self.bidirectional)
        print(identifier, ' Init:', self.init_functions)
        print(identifier, ' Act:', self.act_functions)
    def codify_components(self):
        units = [str(x) for x in self.units_in_layer]
        bidirectional = [str(x) for x in self.bidirectional]
        init_funcs = [str(x) for x in self.init_functions]
        act_funcs = [str(x) for x in self.act_functions]
        return str(self.input_dim) + "_" + str(self.output_dim) + "_" + str(self.rnn_layers) + "_" + ",".join(units) + \
               "_" + ",".join(bidirectional) + "_" + ",".join(init_funcs) + "_" + ",".join(act_funcs)
    
class Network:
    def __init__(self, network_descriptor):
        """
        This class contains the tensorflow definition of the networks (i.e., the "implementation" of the descriptors)
        :param network_descriptor: The descriptor this class is implementing
        """
        self.descriptor = network_descriptor

class MLP(Network):

    def __init__(self, network_descriptor):
        super().__init__(network_descriptor)

    def building(self, x):
        """
        This function creates the MLP's model
        :param _: Convenience
        :return: Generated Keras model representing the MLP
        """
        
        for lay_indx in range(self.descriptor.number_hidden_layers):
            
            x = Dense(self.descriptor.dims[lay_indx], 
                      activation=self.descriptor.act_functions[lay_indx], 
                      kernel_initializer=self.descriptor.init_functions[lay_indx])(x)
            if self.descriptor.dropout:
                x = Dropout(self.descriptor.dropout_probs[lay_indx])(x)
            if self.descriptor.batch_norm:
                x = BatchNormalization()(x)
        
        x = Dense(self.descriptor.output_dim, 
                  activation=self.descriptor.act_functions[self.descriptor.number_hidden_layers],
                  kernel_initializer=self.descriptor.init_functions[self.descriptor.number_hidden_layers])(x)
        if self.descriptor.dropout:
            x = Dropout(self.descriptor.dropout_probs[lay_indx])(x)
        if self.descriptor.batch_norm:
            x = BatchNormalization()(x)
        
        return x


class CNN(Network):

    def __init__(self, network_descriptor):
        super().__init__(network_descriptor)

    def building(self, x):
        """
        Using the filters defined in the initialization function, create the CNN
        :param layer: Input of the network
        :return: Output of the network
        """
        lay_indx = 0
        while lay_indx < self.descriptor.number_hidden_layers:
            
            x = Conv2D(self.descriptor.filters[lay_indx][2],
                       [self.descriptor.filters[lay_indx][0],self.descriptor.filters[lay_indx][1]],
                       strides=[self.descriptor.strides[lay_indx][0], self.descriptor.strides[lay_indx][1]],
                       padding="valid",
                       activation=self.descriptor.act_functions[lay_indx],
                       kernel_initializer=self.descriptor.init_functions[lay_indx])(x)
            
            if self.descriptor.batch_norm:
                x = BatchNormalization()(x)

            lay_indx += 1

            if lay_indx < self.descriptor.number_hidden_layers:
               
                if self.descriptor.layers[lay_indx] == 0:  # If is has average pooling
                    x = AveragePooling2D(pool_size=[self.descriptor.filters[lay_indx][0], self.descriptor.filters[lay_indx][1]],
                                               strides=[self.descriptor.strides[lay_indx][0], self.descriptor.strides[lay_indx][1]],
                                               padding="valid")(x)
                    lay_indx += 1
                elif self.descriptor.layers[lay_indx] == 1: # If it has max pooling
                    x = MaxPooling2D(pool_size=[self.descriptor.filters[lay_indx][0], self.descriptor.filters[lay_indx][1]],
                                           strides=[self.descriptor.strides[lay_indx][0], self.descriptor.strides[lay_indx][1]],
                                           padding="valid")(x)
                    lay_indx += 1
                
        return x


class TCNN(Network):
    """
    Almost identical to CNN
    """
    def __init__(self, network_descriptor):
        super().__init__(network_descriptor)
    
    def building(self, x):
        
        for lay_indx in range(self.descriptor.number_hidden_layers):
            x = Conv2DTranspose(self.descriptor.filters[lay_indx][2],
                                      [self.descriptor.filters[lay_indx][0],self.descriptor.filters[lay_indx][1]],
                                      strides=[self.descriptor.strides[lay_indx][0], self.descriptor.strides[lay_indx][1]],
                                      padding="valid",
                                      activation=self.descriptor.act_functions[lay_indx],
                                      kernel_initializer=self.descriptor.init_functions[lay_indx])(x)

        return x

class RNN(Network):
    def __init__(self, network_descriptor):
        super().__init__(network_descriptor)
        
    def building(self, x):
        for lay_indx in range(self.descriptor.number_hidden_layers - 1):
            
            rnn_layer = self.descriptor.rnn_layers[lay_indx](units=self.descriptor.units_in_layer[lay_indx],
                                                             return_sequences=True,
                                                             activation=self.descriptor.act_functions[lay_indx],
                                                             kernel_initializer=self.descriptor.init_functions[lay_indx]())
            
            if self.descriptor.bidirectional[lay_indx]:
                x = Bidirectional(rnn_layer)(x)
            else:
                x = rnn_layer(x)
        
            if self.descriptor.dropout:
                x = Dropout(self.descriptor.dropout_probs[lay_indx])(x)
            
        # Last layer is the one that decide the type of output: single or seguence
        if len(self.descriptor.output_dim) == 1:
            return_sequence = False
        else:
            return_sequence = True
            
        rnn_layer = self.descriptor.rnn_layers[self.descriptor.number_hidden_layers - 1](
                        units=self.descriptor.units_in_layer[self.descriptor.number_hidden_layers - 1],
                        return_sequences=return_sequence,
                        activation=self.descriptor.act_functions[self.descriptor.number_hidden_layers - 1],
                        kernel_initializer=self.descriptor.init_functions[self.descriptor.number_hidden_layers - 1]())
        
        if self.descriptor.bidirectional[self.descriptor.number_hidden_layers]:
            x = Bidirectional(rnn_layer)(x)
        else:
            x = rnn_layer(x)
            
        x = Dense(self.descriptor.output_dim[-1], activation='softmax')(x)
            
        return x

def calculate_CNN_shape(input_shape, filters, strides, desired_layer):
    if desired_layer == -1:
        return calculate_CNN_shape(input_shape, filters, strides, len(filters))
    if desired_layer == 0:
        return input_shape
    
    filter_size = filters[0]
    stride_size = strides[0]
    output_shape = (np.array(input_shape[:2]) - np.array(filter_size[:2]) + 1) // np.array(stride_size[:2])
    return calculate_CNN_shape(output_shape, filters[1:], strides[1:], desired_layer-1)


def calculate_TCNN_shape(input_shape, filters, strides, desired_layer):
    if desired_layer == -1:
        return calculate_TCNN_shape(input_shape, filters, strides, len(filters))
    if desired_layer == 0:
        return input_shape
    
    filter_size = filters[0]
    stride_size = strides[0]
    output_shape = [input_shape[0] * stride_size[0] + max(filter_size[0] - stride_size[0], 0), 
                    input_shape[1] * stride_size[1] + max(filter_size[1] - stride_size[1], 0)]
    return calculate_TCNN_shape(output_shape, filters[1:], strides[1:], desired_layer-1)