import sys
sys.path.append('../..')

import pandas as pd
import numpy as np

from deatf.evolution import Evolving

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

import tensorflow_datasets as tfds

def test(dataset_name, descriptors=[], eval_func=None, batch_size=150, population=5, 
         generations=10, iters=100, max_num_layers=10, max_num_neurons=20, hyperparameters={},
         is_time_series=False, series_input_width=30, series_label_width=1):
    
    x_train, x_test, x_val, y_train, y_test, y_val, mode = load_dataset(dataset_name, 
            is_time_series=is_time_series, series_input_width=series_input_width, series_label_width=series_label_width)
    
    if not isinstance(y_train[0], float):
        OHEnc = OneHotEncoder()
    
        y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()
        y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
        y_val = OHEnc.fit_transform(np.reshape(y_val, (-1, 1))).toarray()
        
    else:
        y_train = np.reshape(y_train, (-1, 1))
        y_test = np.reshape(y_test, (-1, 1))
        y_val = np.reshape(y_val, (-1, 1))

    input_shape = x_train.shape[1:]
    output_shape = y_val.shape[1:]
    
    if eval_func == None:
        eval_func = select_evaluation(mode)
    
    e = Evolving(evaluation=eval_func, 
			 desc_list=descriptors, 
			 x_trains=[x_train], y_trains=[y_train], 
			 x_tests=[x_val], y_tests=[y_val],
			 n_inputs=[input_shape],
			 n_outputs=[output_shape],
			 batch_size=batch_size,
			 population=population,
			 generations=generations,
			 iters=iters, 
			 max_num_layers=max_num_layers, 
			 max_num_neurons=max_num_neurons,
             hyperparameters=hyperparameters)   
     
    a = e.evolve()
    return a

def select_evaluation(mode):
    
    if mode == "class":
        loss = "XEntropy"
    if mode == "regr":
        loss = "MSE"
        
    return loss

def load_dataset(dataset_name, is_time_series=False, series_input_width=30, series_label_width=1):
    
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
        if is_time_series:
            features, labels = transform_to_time_series(features, labels, series_input_width, series_label_width)
    elif dataset_name in datasets_CNN:
        features, labels, mode = load_CNN(dataset_name)
    else:
        raise ValueError('The dataset \'{}\' is not available'.format(dataset_name))
        
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
    
    dividing_indx = int(X_train.shape[0] * 0.7)

    X_val = X_train[dividing_indx:]
    y_val = y_train[dividing_indx:]
    
    X_train = X_train[:dividing_indx]
    y_train = y_train[:dividing_indx]
    
    return X_train, X_test, X_val, y_train, y_test, y_val, mode

def load_mushrooms():    
    features, labels = load_csv("./datasets/mushroom/mushroom.csv",'\\t', '.',
                                'class',[])
    labels = [int(l) for l in labels]
    return features, labels, 'class'

def load_air_quality():
    features, labels = load_csv("./datasets/air_quality/AirQualityUCI.csv",';', ',',
                                'RH',['Date','Time','Unnamed: 15','Unnamed: 16'])
    
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
    labels = [int(l) for l in labels]
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

    scaler = MinMaxScaler(feature_range=(0, 1))
    features = scaler.fit_transform(features)
    if isinstance(labels[0], float) or isinstance(labels[0], int):
        labels = scaler.fit_transform(labels.reshape(-1, 1)).flatten()
    
    return features, labels

def transform_to_time_series(features, labels, input_width, label_width):
        
    x, y = [], []
    
    for i in range(len(features)-(input_width+label_width)+1):
        x.append(features[i:i+input_width])
    x = np.array(x)
    for i in range(label_width):
        y.append(labels[i+input_width:len(labels)-label_width+i+input_width-1])
    y = np.array(y)
    y = np.transpose(y)

    return x, y
    

if __name__ == "__main__":
    #features, labels, model = load_air_quality()
    #features, labels, model = load_CNN('cifar10')
    X_train, X_test, X_val, y_train, y_test, y_val, mode = load_dataset('air_quality', True, 30, 1)
    #x,y = transform_to_time_series(features, labels, 30, 1)