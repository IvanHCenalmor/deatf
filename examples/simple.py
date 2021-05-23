"""
This is a use case of EvoFlow

In this instance, we require a simple, single-DNN layer classifier for which we specify the predefined loss and fitness function.
"""
import sys
sys.path.append('..')

import numpy as np

from evoflow.network import MLPDescriptor
from evoflow.evolution import Evolving
from evoflow.metrics import ret_evals
from evoflow.data import load_fashion

from sklearn.preprocessing import OneHotEncoder

if __name__ == "__main__":

    x_train, y_train, x_test, y_test, x_val, y_val = load_fashion()

    OHEnc = OneHotEncoder()

    y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()
    y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
    y_val = OHEnc.fit_transform(np.reshape(y_val, (-1, 1))).toarray()

    e = Evolving(evaluation="XEntropy", desc_list=[MLPDescriptor], 
    			 x_trains=[x_train], y_trains=[y_train], x_tests=[x_val], y_tests=[y_val], 
    			 n_inputs=[[28, 28]], n_outputs=[[10]], batch_size=150, 
    			 population=5, generations=3, iters=10, n_layers=10, max_layer_size=20,
                 seed=0, dropout=False, batch_norm=False, ev_alg='mu_plus_lambda',
                 evol_kwargs={'mu':2, 'lambda_':5, 'cxpb':1., "mutpb": 0.},
                 sel = 'best')
    
    a = e.evolve()

    np.save("simple_res_rand.npy", np.array(ret_evals()))
    print(a)
