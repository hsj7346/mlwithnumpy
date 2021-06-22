import numpy as np 
import os
import csv

def load_data(data_path):
    with open(data_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        count = 0
        col_names = []
        data = []
        for line in reader:
            if count == 0:
                col_names.append(line)
                count = count + 1
            else:
                data.append(line)
        attribute_names = col_names[0][:-1]
        features = np.array(data, dtype=float)[:,:-1]
        targets = np.array(data, dtype=float)[:,-1]
        
    return features, targets, attribute_names

def train_test_split(features, targets, fraction):
    if (fraction > 1.0):
        raise ValueError('N cannot be bigger than number of examples!')

    elif (fraction == 1.0):
        train_features = features
        train_targets = targets
        test_features = features
        test_targets = targets
        return train_features,train_targets,test_features,test_targets
    else:
        N = int(features.shape[0] * fraction)
        M = features.shape[0] - N
        features1 = np.random.permutation(features.shape[0])
        N_features = features1[:N]
        M_features = features1[N:]
        train_features = features[N_features,:]
        test_features = features[M_features,:]
        train_targets = targets[N_features].reshape(N,1)
        test_targets = targets[M_features].reshape(M,1)
        return train_features,train_targets,test_features,test_targets

