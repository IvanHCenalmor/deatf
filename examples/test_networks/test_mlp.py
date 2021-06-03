import sys
sys.path.append('../..')

import time

from deatf.network import MLPDescriptor

from aux_functions_testing import test

def test_MLP_all_datasets(eval_func=None, batch_size=150, population=5, 
                      generations=10, iters=100, max_num_layers=10, max_num_neurons=20):
    
    dataset_collection = ['mushrooms', 'air_quality', 'estambul_values', 
                          'forest_fires', 'forest_types', 'parkinsons']

    for dataset in dataset_collection:
        
        print('\nEvaluating the {} dataset with the following configuration:'.format(dataset),
              '\nBatch size:  {}\nPopulation of networks:  {}\nGenerations:  {}'.format(batch_size, population, generations),
              '\nIterations in each network:  {}\nMaximum number of layers:  {}'.format(iters, max_num_layers),
              '\nMaximum number of neurons in each layer: {}'.format(max_num_neurons))
        
        init_time = time.time()
        
        try:
            x = test_MLP(dataset, eval_func=eval_func, batch_size=batch_size, 
                 population=population, generations=generations, iters=iters, 
                 max_num_layers=max_num_layers, max_num_neurons=max_num_neurons)
            print(x)
        except Exception as e:
            print('An error ocurred executing the {} dataset.'.format(dataset))    
            print(e)
    
        print('Time: ', time.time() - init_time)

def test_MLP(dataset_name, eval_func=None, batch_size=150, population=5, 
             generations=10, iters=100, max_num_layers=10, max_num_neurons=20):
    
    
    return test(dataset_name, descriptors=[MLPDescriptor], eval_func=eval_func, 
                batch_size=batch_size, population=population, generations=generations, 
                iters=iters, max_num_layers=max_num_layers, max_num_neurons=max_num_neurons)

if __name__ == "__main__":
    evaluated = test_MLP('parkinsons', batch_size=50, population=10, generations=5, iters=10)
    #test_MLP_all_datasets(batch_size=200, population=2, generations = 2, iters=10)
