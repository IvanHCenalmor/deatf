"""
This is a use case of EvoFlow

We face here an unsupervised problem. We try to reduce the dimensionality of data by using a convolutional autoencoder. For that we
define a CNN that encodes data to a reduced dimension, and a transposed CNN (TCNN) for returning it to its original form.
"""
import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from evoflow.network import ConvDescriptor, TConvDescriptor
from evoflow.evolution import Evolving
from evoflow.data import load_fashion

from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as opt

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]


def eval_cnn_ae(nets, train_inputs, _, batch_size, test_inputs, __, hypers):
    models = {}

    inp = Input(shape=train_inputs["i0"].shape[1:])
    out = nets["n0"].building(inp)
    out = Flatten()(out)
    out = Dense(49)(out)
    out = tf.reshape(out, (-1, 7, 7, 1))
    out = nets["n1"].building(out)
    
    model = Model(inputs=inp, outputs=out)
    
    opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])
    model.compile(loss=tf.losses.mean_squared_error, optimizer=opt, metrics=[])
    
    # As the output has to be the same as the input, the input is passed twice
    model.fit(train_inputs['i0'], train_inputs['i0'], epochs=10, batch_size=batch_size, verbose=0)
            
    models["n0"] = model
                     
    
    pred = models["n0"].predict(test_inputs["i0"])
    res = pred[:, :28, :28, :3] 
        
    if np.isnan(res).any():
        return 288,
    else:
        return mean_squared_error(np.reshape(res, (-1)), np.reshape(test_inputs["i0"], (-1))),


if __name__ == "__main__":

    x_train, y_train, x_test, y_test, x_val, y_val = load_fashion()
    
    x_train = x_train[:10000]
    y_train = y_train[:10000]
    x_test = x_test[:5000]
    y_test = y_test[:5000]
    x_val = x_val[:5000]
    y_val = y_val[:5000]
    
    # We fake a 3 channel dataset by copying the grayscale channel three times.
    x_train = np.expand_dims(x_train, axis=3)/255
    x_train = np.concatenate((x_train, x_train, x_train), axis=3)

    x_test = np.expand_dims(x_test, axis=3)/255
    x_test = np.concatenate((x_test, x_test, x_test), axis=3)

    x_val = np.expand_dims(x_val, axis=3)/255
    x_val = np.concatenate((x_val, x_val, x_val), axis=3)
    
    OHEnc = OneHotEncoder()

    y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()

    y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
    
    y_val = OHEnc.fit_transform(np.reshape(y_val, (-1, 1))).toarray()   
    # Here we define a convolutional-transposed convolutional network combination
    
    e = Evolving(desc_list=[ConvDescriptor, TConvDescriptor], 
                 x_trains=[x_train], y_trains=[y_train], 
                 x_tests=[x_val], y_tests=[y_val], 
                 evaluation=eval_cnn_ae, 
                 batch_size=150, 
                 population=2, 
                 generations=10, 
                 n_inputs=[[28, 28, 3], [7, 7, 1]], 
                 n_outputs=[[49], [28, 28, 3]], 
                 cxp=0, 
                 mtp=1, 
                 hyperparameters = {"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]}, 
                 batch_norm=True, 
                 dropout=True)
    a = e.evolve()
    
    print(a[-1])
    
