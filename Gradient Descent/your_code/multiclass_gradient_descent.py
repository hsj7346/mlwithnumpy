import numpy as np
from your_code import GradientDescent


class MultiClassGradientDescent:

    def __init__(self, loss, regularization=None,
                 learning_rate=0.01, reg_param=0.05):
        self.loss = loss
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.reg_param = reg_param

        self.model = []
        self.classes = None

    def fit(self, features, targets, batch_size=None, max_iter=1000):
        self.classes = np.unique(targets)
        X = self.classes.shape[0]
        for i in range(X):
            model = GradientDescent(self.loss,self.regularization,self.learning_rate,self.reg_param)
            self.model.append(model)
        for i in range(X):
            if len(self.classes) == 2:
                self.model[i].fit(features,targets,batch_size,max_iter)
            else:
                targets1 = np.where(targets == self.classes[i], 2, -2)
                self.model[i].fit(features,targets1,batch_size,max_iter)


    def predict(self, features):
        prediction = np.zeros((features.shape[0],len(self.model)))
        for i in range(len(self.model)):
            self.model[i].confidence(features)
            prediction[:,i] = self.model[i].predict(features)
        if len(self.model) != 2:
            return np.argmax(prediction,axis=1)
        a = np.argmax(prediction,axis=1)
        return np.array([prediction[i,a[i]] for i in range(features.shape[0])])
