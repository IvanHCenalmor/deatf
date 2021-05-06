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
from evolution import Evolving, accuracy_error
from Network import MLPDescriptor, ConvDescriptor

from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]

def eval_cnn(nets, train_inputs, train_outputs, batch_size, test_inputs, test_outputs, hypers):
    models = {}
    
    inp = Input(shape=train_inputs["i0"].shape[1:])
    out = nets["n0"].building(inp)
    out = Flatten()(out)
    out = Dense(20)(out)
    out = nets["n1"].building(out)
    
    model = Model(inputs=inp, outputs=out)

    opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])
    model.compile(loss=tf.nn.softmax_cross_entropy_with_logits, optimizer=opt, metrics=[])
    
    model.fit(train_inputs['i0'], train_outputs['o0'], epochs=10, batch_size=batch_size, verbose=0)
    
    models["n0"] = model
    
    preds = models["n0"].predict(test_inputs["i0"])
    
    res = tf.nn.softmax(preds)

    return accuracy_error(res, test_outputs["o0"]),


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
    
    # Here we indicate that we want a CNN as the first network of the model
    e = Evolving(desc_list=[ConvDescriptor, MLPDescriptor], x_trains=[x_val], y_trains=[y_val], 
                 x_tests=[x_test], y_tests=[y_test], evaluation=eval_cnn, 
                 batch_size=150, population=5, generations=10, n_inputs=[[28, 28, 3], [20]], n_outputs=[[20], [10]], cxp=0.5,
                 mtp=0.5, hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]}, batch_norm=True, dropout=True)
    a = e.evolve()

    print(a[-1])
