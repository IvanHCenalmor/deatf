"""
This is a use case of EvoFlow

In this instance, we require a simple, single-DNN layer classifier for which we specify the predefined loss and fitness function.
"""
import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np

from evoflow.network import MLPDescriptor
from evoflow.evolution import Evolving
from evoflow.data import load_fashion

from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model

def hand_made_tf_mse(target, prediction):
    return tf.reduce_mean(tf.math.squared_difference(target, prediction))

def hand_made_np_mse(target, prediction):
    return np.mean(np.square(target-prediction))

def ae_eval(nets, train_inputs, train_outputs, batch_size, iters, test_inputs, test_outputs, hypers):
    """
    Function for evolving a single individual. No need of the user providing a evaluation function
    :param individual: DEAP individual
    :return: Fitness value
    """
    
    inp = Input(shape=train_inputs["i0"].shape[1:])
    out = Flatten()(inp)
    out = nets['n0'].building(out)
    model = Model(inputs=inp, outputs=out)

    opt = tf.keras.optimizers.Adam(learning_rate=hypers["lrate"])
    model.compile(loss=hand_made_tf_mse, optimizer=opt, metrics=[])
    
    model.fit(train_inputs['i0'], train_inputs['i0'], epochs=iters, batch_size=batch_size, verbose=0)
    
    prediction = model.predict(test_inputs['i0'])
    
    ev = hand_made_np_mse(test_inputs['i0'], prediction)
    
    if isinstance(ev, float):
        ev = (ev,)
        
    return ev

if __name__ == "__main__":

    x_train, _, x_test, _, x_val, _ = load_fashion()

    x_train = np.reshape(x_train, (-1, 784))
    x_test = np.reshape(x_test, (-1, 784))
    x_val = np.reshape(x_val, (-1, 784))

    x_train = x_train/1
    x_test = x_test/1
    x_val = x_val/1

    e = Evolving(evaluation=ae_eval, desc_list=[MLPDescriptor], 
                 x_trains=[x_train], y_trains=[x_train], x_tests=[x_val], y_tests=[x_val], 
                 n_inputs=[[784]], n_outputs=[[784]], batch_size=150, iters=10, 
                 hyperparameters={"lrate": [0.1, 0.5, 1]},
                 population=5, generations=20, iters=100, n_layers=10, max_layer_size=100, seed=0)
    a = e.evolve()

    print(a)
