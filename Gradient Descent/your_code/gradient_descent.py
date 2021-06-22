import numpy as np
from your_code import HingeLoss, SquaredLoss
from your_code import L1Regularization, L2Regularization


class GradientDescent:
    def __init__(self, loss, regularization=None,
                 learning_rate=0.01, reg_param=0.05):
        self.learning_rate = learning_rate

        # Select regularizer
        if regularization == 'l1':
            regularizer = L1Regularization(reg_param)
        elif regularization == 'l2':
            regularizer = L2Regularization(reg_param)
        elif regularization is None:
            regularizer = None
        else:
            raise ValueError(
                'Regularizer {} is not defined'.format(regularization))

        # Select loss function
        if loss == 'hinge':
            self.loss = HingeLoss(regularizer)
        elif loss == 'squared':
            self.loss = SquaredLoss(regularizer)
        else:
            raise ValueError('Loss function {} is not defined'.format(loss))

        self.model = None

    

    def fit(self, features, targets, batch_size=None, max_iter=1000):
        def conv_criteria(prev,curr):
            if np.abs(prev-curr) < 1e-4:
                return True
            else:
                return False
        self.model = np.random.uniform(-0.1,0.1,features.shape[1]+1)
        features1 = np.column_stack((features,np.ones((features.shape[0],1))))
        prev = 0
        if batch_size == None:
            batch_size = features.shape[0]
        i = 0
        while i < max_iter:
            curr = self.loss.forward(features1,self.model,targets)
            if conv_criteria(prev,curr) == False:
                batch = np.random.choice(np.array([i for i in range(features1.shape[0])]), size = batch_size, replace = False, p = None)
                gradient = self.loss.backward(features1[batch],self.model,targets[batch])
                self.model -= self.learning_rate*gradient
                prev = curr
                i += 1
            else:
                return

    def predict(self, features):
        prediction = np.sign(self.confidence(features))
        result = np.where(prediction == 0, 1, prediction)
        return result


    def confidence(self, features):
        features1 = np.column_stack((features,np.ones((features.shape[0],1))))
        return np.dot(features1,self.model)
