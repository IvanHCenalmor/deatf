import sys
sys.path.append('../..')

import tensorflow as tf
import numpy as np
import time

from evoflow.network import TConvDescriptor
from evoflow.evolution import Evolving

from aux_functions_testing import select_evaluation, load_dataset

import tensorflow.keras.optimizers as opt
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]

def test_TCNN_all_datasets(eval_func=None, batch_size=150, population=5, 
                      generations=10, iters=100, n_layers=10, max_layer_size=20):
    
    dataset_collection = ['mnist', 'kmnist', 'cmaterdb', 'fashion_mnist', 'omniglot', 
                          'binary_alpha_digits', 'cifar10', 'rock_paper_scissors']
   
    for dataset in dataset_collection:
        
        print('\nEvaluating the {} dataset with the following configuration:'.format(dataset),
              '\nBatch size:  {}\nPopulation of networks:  {}\nGenerations:  {}'.format(batch_size, population, generations),
              '\nIterations in each network:  {}\nMaximum number of layers:  {}'.format(iters, n_layers),
              '\nMaximum number of neurons in each layer: {}'.format(max_layer_size))

        init_time = time.time()
        
        try:
            x = test_TCNN(dataset, eval_func=eval_func, batch_size=batch_size, 
                 population=population, generations=generations, iters=iters, 
                 n_layers=n_layers, max_layer_size=max_layer_size)
            print(x)
        except Exception as e:
            print('An error ocurred executing the {} dataset.'.format(dataset))    
            print(e)
            
        print('Time: ', time.time() - init_time)

def test_TCNN(dataset_name, eval_func=None, batch_size=150, population=5, 
             generations=10, iters=100, n_layers=10, max_layer_size=20):
    
    x_train, x_test, x_val, _, _, _, mode = load_dataset(dataset_name)
    
    x_train = x_train[:5000]
    x_test = x_test[:2500]
    x_val = x_val[:2500]
    
    train_noise = np.random.normal(size=(x_train.shape[0], 7, 7, 1))
    test_noise = np.random.normal(size=(x_test.shape[0], 7, 7, 1))
    val_noise = np.random.normal(size=(x_val.shape[0], 7, 7, 1))
    
    input_shape = train_noise.shape[1:]
    output_shape = x_train.shape[1:]
    
    if eval_func == None:
        eval_func = select_evaluation(mode)
    
    e = Evolving(evaluation=eval_func, 
			 desc_list=[TConvDescriptor], 
			 x_trains=[train_noise], y_trains=[x_train], 
			 x_tests=[val_noise], y_tests=[x_val],
			 n_inputs=[input_shape],
			 n_outputs=[output_shape],
			 batch_size=batch_size,
			 population=population,
			 generations=generations,
			 iters=iters, 
			 n_layers=n_layers, 
			 max_layer_size=max_layer_size,
             hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]})   
     
    a = e.evolve()
    return a 
        
def eval_tcnn(nets, train_inputs, train_outputs, batch_size, test_inputs, test_outputs, hypers):
    models = {}
    
    inp = Input(shape=train_inputs["i0"].shape[1:])
    out = nets["n0"].building(inp)
    
    model = Model(inputs=inp, outputs=out)
    
    opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])
    model.compile(loss=tf.nn.softmax_cross_entropy_with_logits, optimizer=opt, metrics=[])
    
    model.fit(train_inputs['i0'], train_outputs['o0'], epochs=10, batch_size=batch_size, verbose=0)
    
    models["n0"] = model

    preds = models["n0"].predict(test_inputs["i0"])
    
    res = tf.nn.softmax(preds)

    return mean_squared_error(np.reshape(res, (-1)), np.reshape(test_outputs["o0"], (-1))),
    
    
if __name__ == "__main__":
    test_TCNN('fashion_mnist', eval_func=eval_tcnn, batch_size=150, population=20, 
             generations=15, iters=3, n_layers=10, max_layer_size=20)
    #test_TCNN_all_datasets(eval_func=eval_tcnn, batch_size=150, population=2, generations=2, iters=10)