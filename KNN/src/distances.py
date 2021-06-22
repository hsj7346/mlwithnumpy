import numpy as np 

def euclidean_distances(X, Y):
    M = X.shape[0]
    N = Y.shape[0]
    result = np.zeros((M,N))
    for m in range(M):
        for n in range(N):
            result[m,n] = np.linalg.norm(X[m,:]-Y[n,:], ord = None)
    return result



def manhattan_distances(X, Y):
    M = X.shape[0]
    N = Y.shape[0]
    result = np.zeros((M,N))
    for m in range(M):
        for n in range(N):
            result[m,n] = np.linalg.norm(X[m,:]-Y[n,:], ord = 1)
    return result

