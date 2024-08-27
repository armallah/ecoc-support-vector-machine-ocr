#svm.py

import numpy as np
#CVXOPT for convex optimization and quadratic programming solver..
from cvxopt import matrix, solvers

class SVM:
    
    def __init__(self, w, w0, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.w = None
        self.w0 = None
        self.learning_rate = learning_rate,
        self.lambda_param = lambda_param,
        self.n_iters = n_iters
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.w0 = 0
        
        for i in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.w0) >= 1

        