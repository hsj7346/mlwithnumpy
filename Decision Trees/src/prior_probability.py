import numpy as np

class PriorProbability():
    def __init__(self):
        self.most_common_class = None
        self.true_class = 0
        self.false_class = 0

    def fit(self, features, targets):
        for i in range(targets.shape[0]):
            if targets[i] == 1:
                self.true_class += 1
            else:
                self.false_class += 1
        if self.true_class > self.false_class:
            self.most_common_class = 1
        else:
            self.most_common_class = 0
        

    def predict(self, data):
        predictions = np.zeros((data.shape[0],1))
        for i in range(predictions.shape[0]):
            predictions[i] = self.most_common_class
        return predictions