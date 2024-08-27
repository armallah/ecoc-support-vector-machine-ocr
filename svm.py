# svm.py

import numpy as np

# CVXOPT for convex optimization and quadratic programming solver..
from cvxopt import matrix, solvers


class SVM:

    def __init__(self, C=1.0):
        self.C = C
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_weights = None

        self.w0 = None
        self.w = None
        
    def __linear__kernel_func(self, x1, x2):
        """
        Kernel function - linear kernel
        """
        return np.dot(x1, x2)
    
    def __polynomial_kernel_func(self, x1, x2, degree=3):
        """
        Kernel function - polynomial kernel
        """
        return (np.dot(x1, x2) + 1) ** degree
    
    def __rbf_kernel_func(self, x1, x2, gamma=0.1):
        """
        Kernel function - Radial basis function kernel
        """
        return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

    def __calc_gram_matrix(self, X, kenrel_func=self.__linear__kernel_func):
        """
        Calculate the Gram matrix, X^T * X, for wolfe dual problem
        """
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel_func(X[i], X[j])
        return K

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = np.dot(X[i], X[j])
