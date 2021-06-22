import numpy as np
from src import KMeans
from scipy.stats import multivariate_normal

class GMM():
    def __init__(self, n_clusters, covariance_type):
        self.n_clusters = n_clusters
        allowed_covariance_types = ['spherical', 'diagonal']
        if covariance_type not in allowed_covariance_types:
            raise ValueError(f'covariance_type must be in {allowed_covariance_types}')
        self.covariance_type = covariance_type

        self.means = None
        self.covariances = None
        self.mixing_weights = None
        self.max_iterations = 200

    def fit(self, features):
        # 1. Use your KMeans implementation to initialize the means of the GMM.
        kmeans = KMeans(self.n_clusters)
        kmeans.fit(features)
        self.means = kmeans.means

        # 2. Initialize the covariance matrix and the mixing weights
        self.covariances = self._init_covariance(features.shape[-1])

        # 3. Initialize the mixing weights
        self.mixing_weights = np.random.rand(self.n_clusters)
        self.mixing_weights /= np.sum(self.mixing_weights)

        # 4. Compute log_likelihood under initial random covariance and KMeans means.
        prev_log_likelihood = -float('inf')
        log_likelihood = self._overall_log_likelihood(features)

        # 5. While the log_likelihood is increasing significantly, or max_iterations has
        # not been reached, continue EM until convergence.
        n_iter = 0
        while abs(log_likelihood - prev_log_likelihood) > 1e-4 and n_iter < self.max_iterations:
            prev_log_likelihood = log_likelihood

            assignments = self._e_step(features)
            self.means, self.covariances, self.mixing_weights = (
                self._m_step(features, assignments)
            )

            log_likelihood = self._overall_log_likelihood(features)
            n_iter += 1

    def predict(self, features):
        posteriors = self._e_step(features)
        return np.argmax(posteriors, axis=-1)

    def _e_step(self, features):
        result = np.zeros((features.shape[0],self.n_clusters))
        for i in range(self.n_clusters):
            result[:,i] = self._posterior(features,i)
        return result        

    def _m_step(self, features, assignments):
        means = np.zeros((assignments.shape[1],features.shape[1]))
        weights = np.zeros((assignments.shape[1]))
        covariances = np.zeros((assignments.shape[1],features.shape[1]))
        for i in range(assignments.shape[1]):
            big_gamma = np.sum(assignments[:,i])
            means[i] = np.sum(assignments[:,i].reshape((features.shape[0],1)) * features, axis=0) / big_gamma
            weights[i] = np.mean(assignments[:,i])
            covariances[i] = np.sum(assignments[:,i].reshape((features.shape[0],1)) * np.square(features-means[i]), axis=0)/ big_gamma
        return means,covariances,weights

    def _init_covariance(self, n_features):
        if self.covariance_type == 'spherical':
            return np.random.rand(self.n_clusters)
        elif self.covariance_type == 'diagonal':
            return np.random.rand(self.n_clusters, n_features)

    def _log_likelihood(self, features, k_idx):
        return  np.log(self.mixing_weights[k_idx]) + multivariate_normal.logpdf(features,self.means[k_idx],self.covariances[k_idx]) 

    def _overall_log_likelihood(self, features):
        denom = [
            self._log_likelihood(features, j) for j in range(self.n_clusters)
        ]
        return np.sum(denom)
 
    def _posterior(self, features, k):
        num = self._log_likelihood(features, k)
        denom = np.array([
            self._log_likelihood(features, j)
            for j in range(self.n_clusters)
        ])

        # Below is a useful function for safely computing large exponentials. It's a common
        # machine learning trick called the logsumexp trick:
        #   https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/

        max_value = denom.max(axis=0, keepdims=True)
        denom_sum = max_value + np.log(np.sum(np.exp(denom - max_value), axis=0))
        posteriors = np.exp(num - denom_sum)
        return posteriors
