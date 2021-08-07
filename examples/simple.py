"""
This is the simplest use case of DEATF. 

In this instance, we require a simple DNN, a Multi Layer Perceptron (MLP). Only restrictions
for the evolution have to be established, like maximum number of layers or neurons in the MLP.
As is it the simple case, no evalution function has to be used, a predifined one is used (XEntropy).
Fashion mnist dataset is used, that is why 28x28 is the input size and 10 the output size.
"""
import sys
sys.path.append('..')

import numpy as np


from deatf.auxiliary_functions import load_fashion
from deatf.network import MLPDescriptor
from deatf.evolution import Evolving

from sklearn.preprocessing import OneHotEncoder

if __name__ == "__main__":

    x_train, y_train, x_test, y_test, x_val, y_val = load_fashion()

    OHEnc = OneHotEncoder()

    y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()
    y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
    y_val = OHEnc.fit_transform(np.reshape(y_val, (-1, 1))).toarray()

    e = Evolving(evaluation="XEntropy", desc_list=[MLPDescriptor], compl=False,
                 x_trains=[x_train], y_trains=[y_train], x_tests=[x_val], y_tests=[y_val], 
                 n_inputs=[[28, 28]], n_outputs=[[10]], batch_size=150, iters=10, 
                 population=15, generations=10, max_num_layers=10, max_num_neurons=20,
                 seed=0, dropout=False, batch_norm=False, evol_alg='mu_plus_lambda',
                 evol_kwargs={'mu':10, 'lambda_':15, 'cxpb':0., "mutpb": 1.},
                 sel = 'best')
    
    a = e.evolve()

    print(a)
