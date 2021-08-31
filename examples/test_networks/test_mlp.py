import sys
sys.path.append('../..')

import time

from deatf.network import MLPDescriptor

from aux_functions_testing import test

def test_MLP_all_datasets(eval_func=None, batch_size=150, population=5, 
                      generations=10, iters=100, max_num_layers=10, max_num_neurons=20,
                      evol_alg='mu_plus_lambda', sel='best', lrate=0.01, cxp=0, mtp=1,
                      seed=None, sel_kwargs={}, max_filter=4, max_stride=3):
    """
    Tests the MLP network with all the possible datasets and with the specified parameter selection.

    :param eval_func: Evaluation function for evaluating each network.
    :param batch_size: Batch size of the data during the training of the networks.
    :param population: Number of individuals in the populations in the genetic algorithm.
    :param generations: Number of generations that will be done in the genetic algorithm.
    :param iters: Number of iterations that each network will be trained.
    :param max_num_layers: Maximum number of layers allowed in the networks.
    :param max_num_neurons: Maximum number of neurons allowed in the networks.
    :param max_filter: Maximum size of the filter allowed in the networks.
    :param max_stride: Maximum size of the stride allowed in the networks.
    :param evol_alg: Evolving algorithm that will be used during the genetic algorithm.
    :param sel: Selection method that will be used during the genetic algorithm.
    :param sel_kwargs: Arguments for selection method.
    :param lrate: Learning rate that will be used during training.
    :param cxp: Crossover probability that will be used during the genetic algorithm.
    :param mtp: Mutation probability that will be used during the genetic algorithm.
    :param seed: Seed that will be used in every random method.
    """  

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
                 max_num_layers=max_num_layers, max_num_neurons=max_num_neurons,
                 evol_alg=evol_alg, sel=sel, lrate=lrate, cxp=cxp, mtp=mtp,
                 seed=seed, sel_kwargs=sel_kwargs, max_filter=max_filter, max_stride=max_stride)
            print(x)
        except Exception as e:
            print('An error ocurred executing the {} dataset.'.format(dataset))    
            print(e)
    
        print('Time: ', time.time() - init_time)

def test_MLP(dataset_name, eval_func=None, batch_size=150, population=5, 
             generations=10, iters=100, max_num_layers=10, max_num_neurons=20,
             evol_alg='mu_plus_lambda', sel='best', lrate=0.01, cxp=0, mtp=1,
             seed=None, sel_kwargs={}, max_filter=4, max_stride=3):
    """
    Tests the MLP network with the specified dataset and parameter selection.

    :param dataset_name: Name of the dataset that will be used in the genetic algorithm.
    :param eval_func: Evaluation function for evaluating each network.
    :param batch_size: Batch size of the data during the training of the networks.
    :param population: Number of individuals in the populations in the genetic algorithm.
    :param generations: Number of generations that will be done in the genetic algorithm.
    :param iters: Number of iterations that each network will be trained.
    :param max_num_layers: Maximum number of layers allowed in the networks.
    :param max_num_neurons: Maximum number of neurons allowed in the networks.
    :param max_filter: Maximum size of the filter allowed in the networks.
    :param max_stride: Maximum size of the stride allowed in the networks.
    :param evol_alg: Evolving algorithm that will be used during the genetic algorithm.
    :param sel: Selection method that will be used during the genetic algorithm.
    :param sel_kwargs: Arguments for selection method.
    :param lrate: Learning rate that will be used during training.
    :param cxp: Crossover probability that will be used during the genetic algorithm.
    :param mtp: Mutation probability that will be used during the genetic algorithm.
    :param seed: Seed that will be used in every random method.
    :return: The last generation, a log book (stats) and the hall of fame (the best 
                 individuals found).
    """

    return test(dataset_name, descriptors=[MLPDescriptor], eval_func=eval_func, 
                batch_size=batch_size, population=population, generations=generations, 
                iters=iters, max_num_layers=max_num_layers, max_num_neurons=max_num_neurons,  
                hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]},
                evol_alg=evol_alg, sel=sel, lrate=lrate, cxp=cxp, mtp=mtp,
                seed=seed, sel_kwargs=sel_kwargs, max_filter=max_filter, max_stride=max_stride)

if __name__ == "__main__":
    #evaluated = test_MLP('forest_types', batch_size=50, population=10, generations=5, iters=10)
    test_MLP_all_datasets(batch_size=200, population=2, generations = 2, iters=10)