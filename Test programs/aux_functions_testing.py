import sys
sys.path.append('../')
from evolution import Evolving

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import tensorflow_datasets as tfds

def test(dataset_name, descriptors=[], eval_func=None, batch_size=150, population=5, 
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
    
    if eval_func == None:
        eval_func = select_evaluation(mode)
    
    e = Evolving(evaluation=eval_func, 
			 desc_list=descriptors, 
			 x_trains=[x_train], y_trains=[y_train], 
			 x_tests=[x_test], y_tests=[y_test],
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

def select_evaluation(mode):
    
    if mode == "class":
        loss = "XEntropy"
    if mode == "regr":
        loss = "MSE"
        
    return loss

def load_dataset(dataset_name):
    
    datasets = {'mushrooms': load_mushrooms,
                'air_quality': load_air_quality,
                'estambul_values': load_estambul_values,
                'forest_fires': load_forest_fires,
                'forest_types': load_forest_types,
                'parkinsons': load_parkinsons
                }
    
    datasets_CNN = ['mnist', 'kmnist', 'cmaterdb', 'fashion_mnist', 'omniglot', 
                    'binary_alpha_digits', 'cifar10', 'rock_paper_scissors']
    
    if dataset_name in datasets.keys():
        features, labels, mode = datasets[dataset_name]()
    elif dataset_name in datasets_CNN:
        features, labels, mode = load_CNN(dataset_name)
    else:
        raise ValueError('The dataset \'{}\' is not available'.format(dataset_name))
        
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
    
    return X_train, X_test, y_train, y_test, mode

def load_mushrooms():    
    features, labels = load_csv("./datasets/mushroom/mushroom.csv",'\\t', '.',
                                'class',[])
    return features, labels, 'class'

def load_air_quality():
    features, labels = load_csv("./datasets/air_quality/AirQualityUCI.csv",';', ',',
                                'CO(GT)',['Date','Time','Unnamed: 15','Unnamed: 16'])
    
    labels = list(map(float, labels))
    
    return features, labels, 'regr'

def load_estambul_values():
    features, labels = load_csv("./datasets/estambul_values/data_akbilgic.csv",',', '.',
                                'ISE',['date','Unnamed: 10','Unnamed: 11'])
    return features, labels, 'regr'

def load_forest_fires():
    features, labels = load_csv("./datasets/forest_fires/forestfires.csv",',', '.',
                                'area',['month', 'day'])
    return features, labels, 'regr'

def load_forest_types():
    train_features, train_labels = load_csv("./datasets/forest_types/training.csv",',', '.',
                                            'class',[])
    test_features, test_labels = load_csv("./datasets/forest_types/testing.csv",',', '.',
                                            'class',[])
    
    labels = np.concatenate((train_labels, test_labels), axis=0)
    features = np.concatenate((train_features, test_features), axis=0)
    
    labels = list(map(ord, map(lambda x: x.strip(), labels)))
    
    return features, labels, 'class'

def load_parkinsons():
    features, labels = load_csv("./datasets/parkinsons/parkinsons.csv", ',', ',',
                                'status',['name'])
    return features, labels, 'class'

def load_CNN(dataset_name):
    
    dataset = tfds.load(dataset_name)
    
    features = []
    labels = []
    
    if 'train' in dataset.keys():
        dataset_train = dataset["train"]
        for example in tfds.as_numpy(dataset_train):
            image, label = example['image'], example['label']
            features.append(image)
            labels.append(label)
            
    if 'test' in dataset.keys():
        dataset_test = dataset["test"]    
        for example in tfds.as_numpy(dataset_test):
            image, label = example['image'], example['label']
            features.append(image)
            labels.append(label) 
        
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels, 'class'

def load_csv(data_directory, data_sep, decimal, label_column, removed_columns):
    
    data = pd.read_csv(data_directory, sep=data_sep, decimal=decimal, engine='python')    
    data = data.drop(columns=removed_columns)
    data = data.dropna()

    labels = data[label_column].values

    data = data.drop(columns=label_column)
    features = np.array(data)
    
    return features, labels


if __name__ == "__main__":
    #features, labels, model = load_air_quality()
    features, labels, model = load_CNN('cifar10')
    #X_train, X_test, y_train, y_test, mode = load_dataset('mushrooms')
    pass
        