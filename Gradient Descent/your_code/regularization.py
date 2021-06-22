import numpy as np

class Regularization:

    def __init__(self, reg_param=0.05):
        self.reg_param = reg_param

    def forward(self, w):
        pass

    def backward(self, w):
        pass


class L1Regularization(Regularization):
    """
    L1 Regularization for gradient descent.
    """

    def forward(self, w):
        return self.reg_param * np.sum(np.abs(w[:-1]))

    def backward(self, w):
        result = np.sign(w)
        result[-1] = 0
        return result * self.reg_param


class L2Regularization(Regularization):
    """
    L2 Regularization for gradient descent.
    """

    def forward(self, w):
        return 0.5 * self.reg_param * np.sum(np.square(w[:-1]))

    def backward(self, w):
        result = w
        result[-1] = 0
        return result * self.reg_param

