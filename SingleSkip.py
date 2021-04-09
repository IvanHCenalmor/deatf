"""
This is a use case of EvoFlow

In this instance, we handle a classification problem, which is to be solved by two DNNs combined in a sequential layout.
The problem is the same as the one solved in Sequential.py, only that here a CNN is evolved as the first component of the model.
"""
from data import load_fashion
import tensorflow as tf
import tensorflow.keras.optimizers as opt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import evolution
from Network import MLPDescriptor, ConvDescriptor, CNN

from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Concatenate, UpSampling2D
from tensorflow.keras.models import Model

class SkipCNN(CNN):
    def building(self, x, skip):
        """
        Using the filters defined in the initialization function, create the CNN
        :param layer: Input of the network
        :param skip: Example of how to implement a skip connection
        :return: Output of the network
        """
        skip = (self.descriptor.number_hidden_layers % (skip-2)) + 2
        for lay_indx in range(self.descriptor.number_hidden_layers):
            
            if lay_indx == 0:
                skip_layer = x
                skip_kernel_size = x.shape[1]
            
            if skip == lay_indx:
                actual_kernel_size = x.shape[1]
                x = UpSampling2D((skip_kernel_size,skip_kernel_size))(x)
                x = MaxPooling2D((actual_kernel_size, actual_kernel_size))(x)
                x = Concatenate(axis=-1)([x, skip_layer])
                
            if self.descriptor.layers[lay_indx] == 2:  # If the layer is convolutional
                
                x = Conv2D(self.descriptor.filters[lay_indx][2],
                           [self.descriptor.filters[lay_indx][0],self.descriptor.filters[lay_indx][1]],
                           strides=[self.descriptor.strides[lay_indx][0], self.descriptor.strides[lay_indx][1]],
                           padding="valid",
                           activation=self.descriptor.act_functions[lay_indx],
                           kernel_initializer=self.descriptor.init_functions[lay_indx])(x)

            elif self.descriptor.layers[lay_indx] == 0:  # If the layer is average pooling
                x = AveragePooling2D(pool_size=[self.descriptor.filters[lay_indx][0], self.descriptor.filters[lay_indx][1]],
                                           strides=[self.descriptor.strides[lay_indx][0], self.descriptor.strides[lay_indx][1]],
                                           padding="valid")(x)
            else:
                x = MaxPooling2D(pool_size=[self.descriptor.filters[lay_indx][0], self.descriptor.filters[lay_indx][1]],
                                       strides=[self.descriptor.strides[lay_indx][0], self.descriptor.strides[lay_indx][1]],
                                       padding="valid")(x)
                
        return x

evolution.descs["ConvDescriptor"] = SkipCNN

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]

def train_cnn(nets, train_inputs, train_outputs, batch_size, hypers):

    models = {}
    
    inp = Input(shape=train_inputs["i0"].shape[1:])
    out = nets["n0"].building(inp, hypers["skip"])
    out = Flatten()(out)
    out = Dense(20)(out)
    out = nets["n1"].building(out)
    
    model = Model(inputs=inp, outputs=out)
    
    opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])
    model.compile(loss=tf.nn.softmax_cross_entropy_with_logits, optimizer=opt, metrics=[])
    
    model.fit(train_inputs['i0'], train_outputs['o0'], epochs=10, batch_size=batch_size, verbose=0)
            
    models["n0"] = model

    return models


def eval_cnn(models, inputs, outputs, _):
    
    preds = models["n0"].predict(inputs["i0"])
    
    res = tf.nn.softmax(preds)

    return evolution.accuracy_error(res, outputs["o0"]),


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_fashion()
    # We fake a 3 channel dataset by copying the grayscale channel three times.
    x_train = np.expand_dims(x_train, axis=3)/255
    x_train = np.concatenate((x_train, x_train, x_train), axis=3)

    x_test = np.expand_dims(x_test, axis=3)/255
    x_test = np.concatenate((x_test, x_test, x_test), axis=3)

    OHEnc = OneHotEncoder()

    y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()

    y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
    # Here we indicate that we want a CNN as the first network of the model
    e = evolution.Evolving(loss=train_cnn, desc_list=[ConvDescriptor, MLPDescriptor], 
                           x_trains=[x_train], y_trains=[y_train], x_tests=[x_test], y_tests=[y_test],
                           evaluation=eval_cnn, batch_size=150, population=200, generations=10, 
                           n_inputs=[[28, 28, 3], [20]], n_outputs=[[20], [10]], cxp=0.5, mtp=0.5, 
                           hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2], "skip": range(3, 10)}, 
                           batch_norm=True, dropout=True)
    a = e.evolve()

    print(a[-1])
