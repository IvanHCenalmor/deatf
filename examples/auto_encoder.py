"""
This is a use case of EvoFlow

In this instance, we require a simple, single-DNN layer classifier for which we specify the predefined loss and fitness function.
"""

import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np


from deatf.auxiliary_functions import load_fashion
from deatf.network import MLPDescriptor
from deatf.evolution import Evolving

from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model

def hand_made_tf_mse(target, prediction):
    """
    Calculates the mean squared error by using TensorFlow operators.
    
    :param target: Ground of truth data.
    :param prediction: Predicted data.
    :return: Mean squared error in a Tensor form.
    """
    return tf.reduce_mean(tf.math.squared_difference(target, prediction))

def hand_made_np_mse(target, prediction):
    """
    Calculates the mean squared error by using NumPy operators.
    
    :param target: Ground of truth data.
    :param prediction: Predicted data.
    :return: Mean squared error in a NumPy form.
    """
    return np.mean(np.square(target-prediction))

def ae_eval(nets, train_inputs, train_outputs, batch_size, iters, test_inputs, test_outputs, hypers):
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
                 n_inputs=[[784]], n_outputs=[[784]],
                 hyperparameters={"lrate": [0.1, 0.5, 1]},
                 population=5, generations=20,  batch_size=150, iters=100, 
                 max_num_layers=10, max_num_neurons=100, seed=0)
    a = e.evolve()

    print(a)
