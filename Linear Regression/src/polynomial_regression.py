import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

class PolynomialRegression():
    def __init__(self, degree):
        self.degree = degree
        self.coef = None
    
    def fit(self, features, targets):
        x = np.ones((features.shape[0],(self.degree+1)))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i,j] = np.power(features[i],j)
        x_t = np.transpose(x)
        f_term = np.linalg.inv(np.matmul(x_t,x))
        s_term = np.matmul(f_term,x_t)
        self.coef = np.matmul(s_term,targets)

    def predict(self, features):
        x = np.ones((features.shape[0],(self.degree+1)))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i,j] = np.power(features[i],j)
        pred = np.dot(x,self.coef)
        return pred


    def visualize(self, features, targets):
        y = self.predict(features)
        fig = plt.figure()
        plt.scatter(features,targets)
        plt.plot(features,y)
        plt.title("Model")
        plt.xlabel("features")
        plt.ylabel("targets")
        fig.savefig("polynomial model")
