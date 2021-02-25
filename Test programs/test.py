import sys
sys.path.append('../')

from evolution import Evolving
from Network import MLPDescriptor

from sklearn.preprocessing import OneHotEncoder
import numpy as np

from load_dataset import load_dataset

def test(dataset_name, loss_func=None, eval_func=None, batch_size=150, population=5, 
         generations=10, iters=100, n_layers=10, max_layer_size=20):
    
    x_train, x_test, y_train, y_test, mode = load_dataset(dataset_name)
    
    if not isinstance(y_train[0], float):
        OHEnc = OneHotEncoder()
    
        y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()
        y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
    else:
        y_train = np.reshape(y_train, (-1, 1))
        y_test = np.reshape(y_test, (-1, 1))
        
    input_shape = x_train.shape[1:]
    output_shape = y_train.shape[1:]
    
    if loss_func == None:
        loss_func = select_loss(mode)
    if eval_func == None:
        eval_func = select_evaluation(mode)
    
    e = Evolving(loss=loss_func, 
			 desc_list=[MLPDescriptor], 
			 x_trains=[x_train], y_trains=[y_train], 
			 x_tests=[x_test], y_tests=[y_test], 
			 evaluation=eval_func, 
			 n_inputs=[input_shape],
			 n_outputs=[output_shape],
			 batch_size=batch_size,
			 population=population,
			 generations=generations,
			 iters=iters, 
			 n_layers=n_layers, 
			 max_layer_size=max_layer_size)   
     
    a = e.evolve()
    return a

def select_loss(mode):
    
    if mode == "class":
        loss = "XEntropy"
    if mode == "regr":
        loss = "MSE"
        
    return loss

def select_evaluation(mode):
    
    if mode == "class":
        evaluation = "Accuracy_error"
    if mode == "regr":
        evaluation = "MSE"
        
    return evaluation

def test_all_datasets(loss_func=None, eval_func=None, batch_size=150, population=5, 
                      generations=10, iters=100, n_layers=10, max_layer_size=20):
    
    dataset_collection = ['mushrooms', 'air_quality', 'estambul_values', 
                          'forest_fires', 'forest_types', 'parkinsons']

    for dataset in dataset_collection:
        
        print('\nEvaluating the {} dataset with the following configuration:'.format(dataset),
              '\nBatch size:  {}\nPopulation of networks:  {}\nGenerations:  {}'.format(batch_size, population, generations),
              '\nIterations in each network:  {}\nMaximum number of layers:  {}'.format(iters, n_layers),
              '\nMaximum number of neurons in each layer: {}'.format(max_layer_size))
        
        try:
            x = test(dataset, loss_func=None, eval_func=None, batch_size=batch_size, 
                 population=population, generations=generations, iters=iters, 
                 n_layers=n_layers, max_layer_size=max_layer_size)
            #print(x)
        except:
            print('An error ocurred executing the {} dataset.'.format(dataset))        
    
if __name__ == "__main__":
    #test('forest_types')
    test_all_datasets(batch_size=200, population=2, generations = 2, iters=10)