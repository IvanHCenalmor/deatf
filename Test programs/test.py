import sys
sys.path.append('../')

from evolution import Evolving
from Network import MLPDescriptor
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from load_dataset import load_dataset

def test(dataset_name):
    
    x_train, x_test, y_train, y_test = load_dataset(dataset_name)
    
    OHEnc = OneHotEncoder()

    y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()

    y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()

    
    e = Evolving(loss="XEntropy", 
			 desc_list=[MLPDescriptor], 
			 x_trains=[x_train], y_trains=[y_train], 
			 x_tests=[x_test], y_tests=[y_test], 
			 evaluation="Accuracy_error", 
			 n_inputs=(22), 
			 n_outputs=[1], 
			 batch_size=150, 
			 population=30, 
			 generations=10, 
			 iters=100, 
			 n_layers=10, 
			 max_layer_size=20)
    a = e.evolve()
    print(a)
    
test('mushrooms')