import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(dataset_name):
    
    datasets = {'mushrooms': load_mushrooms,
                'air_quality': load_air_quality,
                'estambul_values': load_estambul_values,
                'forest_fires': load_forest_fires,
                'forest_types': load_forest_types,
                'parkinsons': load_parkinsons
                }
    
    features, labels, mode = datasets[dataset_name]()
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
    
    return X_train, X_test, y_train, y_test, mode

def load_mushrooms():    
    features, labels = load_csv("./mushroom/mushroom.csv",'\\t', '.',
                                'class',[])
    return features, labels, 'class'

def load_air_quality():
    features, labels = load_csv("./air_quality/AirQualityUCI.csv",';', ',',
                                'CO(GT)',['Date','Time','Unnamed: 15','Unnamed: 16'])
    
    labels = list(map(float, labels))
    
    return features, labels, 'regr'

def load_estambul_values():
    features, labels = load_csv("./estambul_values/data_akbilgic.csv",',', '.',
                                'ISE',['date','Unnamed: 10','Unnamed: 11'])
    return features, labels, 'regr'

def load_forest_fires():
    features, labels = load_csv("./forest_fires/forestfires.csv",',', '.',
                                'area',['month', 'day'])
    return features, labels, 'regr'

def load_forest_types():
    train_features, train_labels = load_csv("./forest_types/training.csv",',', '.',
                                            'class',[])
    test_features, test_labels = load_csv("./forest_types/testing.csv",',', '.',
                                            'class',[])
    
    labels = np.concatenate((train_labels, test_labels), axis=0)
    features = np.concatenate((train_features, test_features), axis=0)
    
    labels = list(map(ord, map(lambda x: x.strip(), labels)))
    
    return features, labels, 'class'

def load_parkinsons():
    features, labels = load_csv("./parkinsons/parkinsons.csv", ',', ',',
                                'status',['name'])
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
    features, labels, model = load_forest_types()
    #X_train, X_test, y_train, y_test, mode = load_dataset('mushrooms')