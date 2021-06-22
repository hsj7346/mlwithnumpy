import numpy as np 
from .distances import euclidean_distances, manhattan_distances

class KNearestNeighbor():    
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        self.n_neighbors = n_neighbors

        if distance_measure == 'euclidean':
            self.distance_measure = euclidean_distances
        elif distance_measure == 'manhattan':
            self.distance_measure = manhattan_distances
        else:
            raise TypeError("INVALID DISTANCE MEASURE")
        
        def mode(array):
                newarray = np.unique(array, return_counts=True)
                result = newarray[0][np.argmax(newarray[1])]
                return result

        if aggregator == "mean":
            self.agg = np.mean
        elif aggregator == "mode":
            self.agg = mode
        elif aggregator == "median":
            self.agg = np.median
        else:
            raise TypeError("INVALID AGGREGATOR")

        self.features = None
        self.targets = None


    def fit(self, features, targets):
        self.features = features
        self.targets = targets

        

    def predict(self, features, ignore_first=False):
        trained = np.argsort(self.distance_measure(features, self.features), axis = 1)
        if ignore_first == False:
            trained_k = trained[:,:self.n_neighbors]
        else:
            trained_k = trained[:,1:self.n_neighbors+1]
        result = []
        for i in range(features.shape[0]):
            knn_target = []
            for j in range(trained_k.shape[1]):
                knn_target.append(self.targets[trained_k[i,j],:])
            knn_target = np.array(knn_target).reshape(trained_k.shape[1],self.targets.shape[1])
            predict_result = [self.agg(knn_target[:,k]) for k in range(knn_target.shape[1])]
            result.append(predict_result)
        return np.array(result).reshape(features.shape[0],self.targets.shape[1])


