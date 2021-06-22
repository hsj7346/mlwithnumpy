import numpy as np

class KMeans():
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.means = None
        self.labels = None

    # My Code from KNN HW
    def euclidean_distances(self, features, means):
        M = features.shape[0]
        N = means.shape[0]
        distance = np.zeros((M,N))
        for m in range(M):
            for n in range(N):
                distance[m,n] = np.linalg.norm(features[m,:]-means[n,:], ord = None)
        return distance

    def nearest_cl(self, distance):
        labels = np.argmin(distance,axis=1)
        return labels

    def update_mean(self, features, labels):
        for i in range(self.n_clusters):
            X = []
            for j in range(labels.shape[0]):
                if labels[j] != i:
                    pass
                else:
                    X.append(j)
            if X == []:
                pass
            else:
                temp = features[X]
                self.means[i] = np.mean(temp,axis=0)

        
    def update_assignment(self, features):
        new_eu = self.euclidean_distances(features, self.means)
        labels = self.nearest_cl(new_eu)
        self.labels = labels
            

    def fit(self, features):
        old_mean = np.random.rand(self.n_clusters,features.shape[1])
        self.means = features[np.random.randint(0,features.shape[0], size=self.n_clusters)]
        self.update_assignment(features)
        while np.array_equal(self.means, old_mean) == False:
            old_mean = self.means.copy()
            self.update_mean(features,self.labels)
            self.update_assignment(features)
            
            


    def predict(self, features):
        distance = self.euclidean_distances(features,self.means)
        return self.nearest_cl(distance)
