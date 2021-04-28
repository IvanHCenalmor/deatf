import sys
sys.path.append('../')

from evolution import Evolving
from Network import MLPDescriptor, ConvDescriptor

from sklearn.preprocessing import OneHotEncoder
import numpy as np

from evolution import accuracy_error

import tensorflow as tf
import tensorflow.keras.optimizers as opt
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model

from load_dataset import load_dataset

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]

def test(dataset_name, descriptors=[], loss_func=None, eval_func=None, batch_size=150, population=5, 
         generations=10, iters=100, n_layers=10, max_layer_size=20, hyperparameters={}):
    
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
			 desc_list=descriptors, 
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
			 max_layer_size=max_layer_size,
             hyperparameters=hyperparameters)   
     
    a = e.evolve()
    return a

def test_MLP(dataset_name, loss_func=None, eval_func=None, batch_size=150, population=5, 
             generations=10, iters=100, n_layers=10, max_layer_size=20):
    
    return test(dataset_name, [MLPDescriptor], loss_func, eval_func, batch_size, 
                population, generations, iters, n_layers, max_layer_size)


def train_cnn(nets, train_inputs, train_outputs, batch_size, hypers):
    models = {}
    
    inp = Input(shape=train_inputs["i0"].shape[1:])
    out = nets["n0"].building(inp)
    out = Flatten()(out)
    out = Dense(train_outputs["o0"].shape[1])(out)
    
    model = Model(inputs=inp, outputs=out)

    opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])
    model.compile(loss=tf.nn.softmax_cross_entropy_with_logits, optimizer=opt, metrics=[])
    
    model.fit(train_inputs['i0'], train_outputs['o0'], epochs=10, batch_size=batch_size, verbose=0)
    
    models["n0"] = model
    
    return models


def eval_cnn(models, inputs, outputs, _):
    
    preds = models["n0"].predict(inputs["i0"])
    
    res = tf.nn.softmax(preds)

    return accuracy_error(res, outputs["o0"]),
    
def test_CNN(dataset_name, loss_func=None, eval_func=None, batch_size=150, population=5, 
             generations=10, iters=100, n_layers=10, max_layer_size=20):
    
    return test(dataset_name, [ConvDescriptor], train_cnn, eval_cnn, batch_size, 
                population, generations, iters, n_layers, max_layer_size,  
                hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]})
    
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

import time

def test_MLP_all_datasets(loss_func=None, eval_func=None, batch_size=150, population=5, 
                      generations=10, iters=100, n_layers=10, max_layer_size=20):
    
    dataset_collection = ['mushrooms', 'air_quality', 'estambul_values', 
                          'forest_fires', 'forest_types', 'parkinsons']

    for dataset in dataset_collection:
        
        print('\nEvaluating the {} dataset with the following configuration:'.format(dataset),
              '\nBatch size:  {}\nPopulation of networks:  {}\nGenerations:  {}'.format(batch_size, population, generations),
              '\nIterations in each network:  {}\nMaximum number of layers:  {}'.format(iters, n_layers),
              '\nMaximum number of neurons in each layer: {}'.format(max_layer_size))
        
        init_time = time.time()
        
        try:
            x = test_MLP(dataset, loss_func=None, eval_func=None, batch_size=batch_size, 
                 population=population, generations=generations, iters=iters, 
                 n_layers=n_layers, max_layer_size=max_layer_size)
            #print(x)
        except Exception as e:
            print('An error ocurred executing the {} dataset.'.format(dataset))    
            print(e)
            print(x)
    
        print('Time: ', time.time() - init_time)
        
def test_CNN_all_datasets(loss_func=None, eval_func=None, batch_size=150, population=5, 
                      generations=10, iters=100, n_layers=10, max_layer_size=20):
    
    dataset_collection = ['mnist', 'kmnist', 'cmaterdb', 'fashion_mnist', 'omniglot', 
                          'binary_alpha_digits', 'cifa10', 'rock_paper_scissors']

    for dataset in dataset_collection:
        
        print('\nEvaluating the {} dataset with the following configuration:'.format(dataset),
              '\nBatch size:  {}\nPopulation of networks:  {}\nGenerations:  {}'.format(batch_size, population, generations),
              '\nIterations in each network:  {}\nMaximum number of layers:  {}'.format(iters, n_layers),
              '\nMaximum number of neurons in each layer: {}'.format(max_layer_size))

        init_time = time.time()
        
        try:
            x = test_CNN(dataset, loss_func=None, eval_func=None, batch_size=batch_size, 
                 population=population, generations=generations, iters=iters, 
                 n_layers=n_layers, max_layer_size=max_layer_size)
            #print(x)
        except Exception as e:
            print('An error ocurred executing the {} dataset.'.format(dataset))    
            print(e)
            print(x)
    
        print('Time: ', time.time() - init_time)
        
    
if __name__ == "__main__":
    #test_MLP('mushrooms', batch_size=150, population=20, 
    #     generations=10, iters=100, n_layers=10, max_layer_size=20)
    #x = test_CNN('kmnist', loss_func=None, eval_func=None, batch_size=200, 
    #             population=2, generations=2, iters=10, 
    #             n_layers=10, max_layer_size=20)
    #test_MLP_all_datasets(batch_size=200, population=2, generations = 2, iters=10)
    test_CNN_all_datasets(batch_size=150, population=2, generations=2, iters=10)