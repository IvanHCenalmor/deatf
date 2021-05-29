import sys
sys.path.append('../..')

import tensorflow as tf
import numpy as np
import time

from evoflow.metrics import accuracy_error
from evoflow.network import RNNDescriptor
from evoflow.evolution import Evolving

from aux_functions_testing import load_dataset, select_evaluation

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as opt

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]

def test_RNN_all_datasets(eval_func=None, is_time_series=True,series_input_width=30, series_label_width=1, 
                      batch_size=150, population=5, generations=10, iters=100, n_layers=10, max_layer_size=20,
                      hyperparameters={}):
    
    dataset_collection = ['mnist', 'kmnist', 'air_quality', 'estambul_values']
   
    for dataset in dataset_collection:
        
        print('\nEvaluating the {} dataset with the following configuration:'.format(dataset),
              '\nBatch size:  {}\nPopulation of networks:  {}\nGenerations:  {}'.format(batch_size, population, generations),
              '\nIterations in each network:  {}\nMaximum number of layers:  {}'.format(iters, n_layers),
              '\nMaximum number of neurons in each layer: {}'.format(max_layer_size))

        init_time = time.time()
        
        try:
            x = test_RNN(dataset, is_time_series=is_time_series, 
                 series_input_width=series_input_width, series_label_width=series_label_width,
                 eval_func=eval_func, batch_size=batch_size, 
                 population=population, generations=generations, iters=iters, 
                 n_layers=n_layers, max_layer_size=max_layer_size,
                 hyperparameters=hyperparameters)
            print(x)
        except Exception as e:
            print('An error ocurred executing the {} dataset.'.format(dataset))    
            print(e)
            
        print('Time: ', time.time() - init_time)

def test_RNN(dataset_name, is_time_series=True,
             series_input_width=30, series_label_width=1,
             eval_func=None, batch_size=150, population=5, 
             generations=10, iters=100, n_layers=10, max_layer_size=20, 
             hyperparameters={}):
    
    x_train, x_test, x_val, y_train, y_test, y_val, mode = load_dataset(dataset_name,is_time_series=is_time_series, 
                                        series_input_width=series_input_width, series_label_width=series_label_width)
    
    # Shape of logits must be 2
    if len(x_train.shape[1:]) == 3:
        x_train = x_train.reshape(x_train.shape[:-1])
        x_test = x_test.reshape(x_test.shape[:-1])
        x_val = x_val.reshape(x_val.shape[:-1])
    elif len(x_train.shape[1:]) == 1:
        x_train = x_train.reshape(list(x_train.shape) + [1])   
        x_test = x_test.reshape(list(x_test.shape) + [1])   
        x_val = x_val.reshape(list(x_val.shape) + [1])   

    input_shape = list(x_train.shape[1:])

    if len(y_train.shape) == 1:
        output_shape = [y_train.max() + 1]
    else:
        output_shape = y_train.shape[1:]
        
    if eval_func == None:
        eval_func = select_evaluation(mode)
    
    e = Evolving(evaluation=eval_func, 
			 desc_list=[RNNDescriptor], 
			 x_trains=[x_train], y_trains=[y_train], 
			 x_tests=[x_val], y_tests=[y_val],
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
        
def eval_rnn(nets, train_inputs, train_outputs, batch_size, test_inputs, test_outputs, hypers):
    models = {}

    inp = Input(shape=train_inputs["i0"].shape[1:])
    out = nets["n0"].building(inp)
    if len(train_outputs["o0"].shape) == 1:
        out = Dense(10, activation='softmax')(out)
    else:    
        out = Dense(1)(out)
    
    model = Model(inputs=inp, outputs=out)
    #model.summary()
    opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])

    if len(train_outputs["o0"].shape) == 1:
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt, metrics=[])
    else:
        model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=opt, metrics=[])
        
    model.fit(train_inputs['i0'], train_outputs['o0'], epochs=3, batch_size=batch_size, verbose=0)
            
    models["n0"] = model
                     
    preds = models["n0"].predict(test_inputs["i0"])
    
    if len(train_outputs["o0"].shape) == 1:
        res = tf.nn.softmax(preds)
        ev = accuracy_error(res, test_outputs["o0"])
    else:
        if np.isnan(preds).any():
            ev = 288
        else:
            ev = tf.keras.losses.mean_squared_error(preds, test_outputs["o0"])
            ev = np.sqrt(np.mean(ev))
        
    return ev, 
    
if __name__ == "__main__":
    #evaluated = test_RNN('estambul_values', is_time_series=True, series_input_width=50, series_label_width=1,
    #                     eval_func=eval_rnn, batch_size=300, population=4, 
    #                     generations=5, iters=3, n_layers=10, max_layer_size=20,                  
    #                     hyperparameters = {"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]})
    test_RNN_all_datasets(is_time_series=True, series_input_width=50, series_label_width=1,
                          eval_func=eval_rnn, batch_size=150, population=2, 
                          generations=2, iters=10, n_layers=10, max_layer_size=20,                  
                          hyperparameters = {"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]})