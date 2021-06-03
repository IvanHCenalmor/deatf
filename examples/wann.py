"""
This is a use case of EvoFlow

In this instance, we handle a classification problem, which is to be solved by two DNNs combined in a sequential layout.
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
import tensorflow.keras.optimizers as opt
from sklearn.preprocessing import OneHotEncoder

evals = []

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]

"""
This is not a straightforward task as we need to "place" the models in the sequential order.
For this, we need to:
1- Tell the model the designed arrangement.
2- Define the training process.
3- Implement a fitness function to test the models.
"""


def eval_wann(nets, train_inputs, train_outputs, batch_size, iters, test_inputs, test_outputs, hypers):
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
    
    model = Model(inputs=inp, outputs=out)
    
    parameters = 0
    for layer in model.layers:
        if layer.get_weights() != []:
            parameters += np.prod(layer.get_weights()[0].shape)
    
    string = hypers["start"]
    while len(string) < parameters:
        string = string.replace("0", hypers["p1"])
        string = string.replace("1", hypers["p2"])
    aux = 0
    ls = []
    for layer in model.layers:
        lay = []
        if layer.get_weights() != []:
            if len(layer.get_weights()[0].shape) == 2:
                for i in range(layer.get_weights()[0].shape[0]):
                    lay += [[int(i) for i in string[aux:aux+layer.get_weights()[0].shape[1]]]]
                    aux += layer.get_weights()[0].shape[1]
                lay = np.array(lay)
                lay = np.where(lay == 0, hypers["weight1"], lay)
                lay = np.where(lay == 1, hypers["weight2"], lay)
                lay = [lay, layer.get_weights()[1]]
        ls += [lay]
    
    for i, layer in enumerate(model.layers):  # Este for asigna el valor a todos los pesos
        if ls[i]:
            layer.set_weights(ls[i])

    models["n0"] = model

    global evals
    
    preds = models["n0"].predict(test_inputs["i0"])
    
    res = tf.nn.softmax(preds)
    
    res = np.argmax(res, axis=1)
    res = 1 - np.sum(np.argmax(test_outputs["o0"], axis=1) == res) / res.shape[0]

    if len(evals) % 10000 == 0:
        np.save("temp_evals.npy", np.array(evals))
    evals += [res]

    return res,


if __name__ == "__main__":

    x_train, y_train, x_test, y_test, x_val, y_val = load_fashion()
    
    x_train = x_train/255
    x_test = x_test/255
    x_val = x_val/255
    
    y_train = np.array([0 if x <= 4 else 1 for x in y_train])
    y_test = np.array([0 if x <= 4 else 1 for x in y_test])
    y_val = np.array([0 if x <= 4 else 1 for x in y_val])

    OHEnc = OneHotEncoder()

    y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()
    y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
    y_val = OHEnc.fit_transform(np.reshape(y_val, (-1, 1))).toarray()

    e = Evolving(evaluation=eval_wann, desc_list=[MLPDescriptor], 
                 x_trains=[x_train], y_trains=[y_train], x_tests=[x_val], y_tests=[y_val],
                 batch_size=150, population=10, generations=100, iters=10, 
                 n_inputs=[[28, 28]], n_outputs=[[2]], cxp=0, mtp=1,
                 batch_norm=False, dropout=False, 
                 hyperparameters={"weight1": np.arange(-2, 2, 0.5), "weight2": np.arange(-2, 2, 0.5), 
                                  "start": ["0", "1"], "p1": ["01", "10"],
                                  "p2": ["001", "010", "011", "101", "110", "100"]})  # Los pesos, que también evolucionan
    a = e.evolve()
    print(a[-1])
