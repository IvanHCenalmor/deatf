"""
This is a use case of EvoFlow
"""
import sys
sys.path.append('..')

import tensorflow as tf

from evoflow.metrics import accuracy_error
from evoflow.network import RNNDescriptor
from evoflow.evolution import Evolving
from evoflow.data import load_fashion

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as opt


optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]


def eval_rnn(nets, train_inputs, train_outputs, batch_size, iters, test_inputs, test_outputs, hypers):
    """
    
    :param nets:
    :param train_inputs:
    :param train_outputs:
    :param batch_size:
    :param iters:
    :param test_inputs:
    :param test_outputs:
    :param hypers:
    """
  
    models = {}

    inp = Input(shape=train_inputs["i0"].shape[1:])
    out = nets["n0"].building(inp)
    out = Dense(10, activation='softmax')(out) # Aa they are probability distributions, they have to be bewteen 0 an 1
    model = Model(inputs=inp, outputs=out)
    
    model.summary()
    
    opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt, metrics=[])
    
    model.fit(train_inputs['i0'], train_outputs['o0'], epochs=iters, batch_size=batch_size, verbose=0)
            
    models["n0"] = model
                     
    preds = models["n0"].predict(test_inputs["i0"])
    
    res = tf.nn.softmax(preds)

    return accuracy_error(res, test_outputs["o0"]),

if __name__ == "__main__":

    x_train, y_train, x_test, y_test, x_val, y_val = load_fashion()

    # Normalize the input data
    x_train = x_train/255
    x_test = x_test/255
    x_val = x_val/255
    
    # Here we define a convolutional-transposed convolutional network combination
    e = Evolving(evaluation=eval_rnn, 
                 desc_list=[RNNDescriptor], 
                 x_trains=[x_train], y_trains=[y_train], 
                 x_tests=[x_val], y_tests=[y_val], 
                 batch_size=150, population=2, generations=10, iters=10, 
                 n_inputs=[[28, 28]], n_outputs=[[10]], 
                 cxp=0, mtp=1, 
                 hyperparameters = {"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]}, 
                 batch_norm=True, dropout=True,
                 ev_alg='mu_plus_lambda')

    a = e.evolve()

    print(a[-1])
