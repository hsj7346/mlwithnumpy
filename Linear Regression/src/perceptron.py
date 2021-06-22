import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def transform_data(features):
    transformed_features = features
    for i in range(features.shape[0]):
        transformed_features[i][0] = (features[i][0]**2+features[i][1]**2)**0.5
        transformed_features[i][1] = np.arctan((features[i][1])/(features[i][0]))
    return transformed_features

class Perceptron():
    def __init__(self, max_iterations=200):
        self.max_iterations = max_iterations
        self.model = None

    def fit(self, features, targets):
        self.model = np.random.rand(features.shape[1]+1)
        i = 1
        arr_1 = np.ones((features.shape[0],1))
        x = np.hstack((arr_1,features))
        while i < self.max_iterations and self.objective_complete(x,targets) == False:
            for j in range(x.shape[0]):
                obj_fun = np.dot(np.transpose(self.model),x[j]) * targets[j]
                if obj_fun < 0:
                    self.model = self.model + x[j]*targets[j]
            i += 1
            
    def objective_complete(self, features, targets):
        result = np.zeros(features.shape[0])
        for i in range(features.shape[0]):
            result[i] = np.dot(np.transpose(self.model),features[i]) * targets[i]
        for j in result:
            if j < 0:
                return False
        return True

    def predict(self, features):
        result = []
        arr_1 = np.ones((features.shape[0],1))
        x = np.hstack((arr_1,features))
        for i in x:
            if np.dot(self.model,i) < 0:
                result.append(-1)
            else:
                result.append(1)
        return np.array(result)

    def visualize(self, features, targets):
        fig = plt.figure()
        plt.scatter(features[:, 0], features[:, 1], c=targets)
        plt.title("Model")
        plt.xlabel("features")
        plt.ylabel("targets")
        fig.savefig("perceptron model")
