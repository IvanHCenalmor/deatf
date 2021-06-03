"""
This is a use case of EvoFlow

In this instance, we handle a classification problem, which is to be solved by two DNNs combined in a sequential layout.
"""
import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np

from deatf.metrics import accuracy_error
from deatf.network import MLPDescriptor
from deatf.evolution import Evolving
from deatf.data import load_fashion

from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as opt
from sklearn.preprocessing import OneHotEncoder

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]

"""
This is not a straightforward task as we need to "place" the models in the sequential order.
For this, we need to:
1- Tell the model the designed arrangement.
2- Define the training process.
3- Implement a fitness function to test the models.
"""


def eval_sequential(nets, train_inputs, train_outputs, batch_size, iters, test_inputs, test_outputs, hypers):
    """
    This function takes care of arranging the model and training it. It is used by the evolutionary internally,
    and always is provided with the same parameters
    :param nets: Dictionary with the Networks ("n0", "n1", ..., "nm", in the same order as they have been requested in the *desc_list* parameter)
    :param train_inputs: Data to be used for training
    :param train_outputs: Data to be used for training
    :param batch_size: Batch_size to be used when training. It is not mandatory to use it
    :param hypers: Optional hyperparameters being evolved in case they were defined for evolution (in this case we also evolve optimizer selection and learning rate)
    :return: A dictionary with the tf layer which makes the predictions
    """
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
    out = Flatten()(inp)
    out = nets["n0"].building(out)
    out = nets["n1"].building(out)

    model = Model(inputs=inp, outputs=out)
    
    opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])
    
    model.compile(loss=tf.nn.softmax_cross_entropy_with_logits, optimizer=opt, metrics=[])
    
    model.fit(train_inputs['i0'], train_outputs['o0'], epochs=iters, batch_size=batch_size, verbose=0)

    models["n0"] = model

    pred = models['n0'].predict(test_inputs['i0'])
        
    res = tf.nn.softmax(pred)

    return accuracy_error(res, test_outputs["o0"]),


if __name__ == "__main__":

    x_train, y_train, x_test, y_test, x_val, y_val = load_fashion()

    OHEnc = OneHotEncoder()

    y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()
    y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
    y_val = OHEnc.fit_transform(np.reshape(y_val, (-1, 1))).toarray()
    
    # When calling the function, we indicate the training function, what we want to evolve (two MLPs), input and output data for training and
    # testing, fitness function, batch size, population size, number of generations, input and output dimensions of the networks, crossover and
    # mutation probability, the hyperparameters being evolved (name and possibilities), and whether batch normalization and dropout should be
    # present in evolution
    e = Evolving(evaluation=eval_sequential, desc_list=[MLPDescriptor, MLPDescriptor], 
                 x_trains=[x_train], y_trains=[y_train], x_tests=[x_val], y_tests=[y_val], 
                 batch_size=150, population=10, generations=10, iters=10, 
                 n_inputs=[[28, 28], [10]], n_outputs=[[10], [10]], cxp=0.5, mtp=0.5, 
                 hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]},
                 batch_norm=False, dropout=False)
    a = e.evolve()
    print(a[-1])
