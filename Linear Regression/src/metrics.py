import numpy as np

def mean_squared_error(estimates, targets):
    n = estimates.shape[0]
    return (1/n) * (np.sum(np.square(targets-estimates)))