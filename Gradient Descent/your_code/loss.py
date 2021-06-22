import numpy as np


class Loss:

    def __init__(self, regularization=None):
        self.regularization = regularization

    def forward(self, X, w, y):
        pass

    def backward(self, X, w, y):
        pass


class SquaredLoss(Loss):
    """
    The squared loss function.
    """

    def forward(self, X, w, y):
        calc = np.zeros(X.shape[0])
        for i in range(calc.shape[0]):
            calc[i] = 0.5 * np.square((y[i] - np.dot(X[i],w)))
        loss = np.mean(calc)
        if self.regularization == None:
            return loss
        else:
            return loss + self.regularization.forward(w)

    def backward(self, X, w, y):
        gradient = -(1/X.shape[0]) * np.dot(np.transpose(X), (y - np.dot(X,w)))
        if self.regularization == None:
            return gradient
        else:
            return gradient + self.regularization.backward(w)


class HingeLoss(Loss):
    """
    The hinge loss function.

    https://en.wikipedia.org/wiki/Hinge_loss
    """

    def forward(self, X, w, y):
        calc = np.zeros(X.shape[0])
        for i in range(calc.shape[0]):
            calc[i] = max(0, 1 - (y[i] * np.dot(X[i], w)))
        loss = np.mean(calc)
        if self.regularization == None:
            return loss
        else:
            return loss + self.regularization.forward(w)

    def backward(self, X, w, y):
        gradient = np.zeros(X.shape)
        for i in range(X.shape[0]):
            calc = 1 - y[i] * np.dot(X[i],w)
            if calc > 0:
                gradient[i] = -y[i] * X[i]
            else:
                gradient[i] = 0
        gradient = np.mean(gradient,axis=0)
        if self.regularization == None:
            return gradient
        else:
            return gradient + self.regularization.backward(w)


class ZeroOneLoss(Loss):

    def forward(self, X, w, y):
        predictions = (X @ w > 0.0).astype(int) * 2 - 1
        loss = np.sum((predictions != y).astype(float)) / len(X)
        if self.regularization:
            loss += self.regularization.forward(w)
        return loss

    def backward(self, X, w, y):
        # This function purposefully left blank
        raise ValueError('No need to use this function for the homework :p')
