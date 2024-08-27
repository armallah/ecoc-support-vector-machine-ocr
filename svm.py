# svm.py

import numpy as np

# CVXOPT for convex optimization and quadratic programming solver..
from cvxopt import matrix, solvers


class SVM:

    def __init__(self, C=1.0):
        self.C = C
        self.sv = None
        self.sv_label = None
        self.sv_W = None

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

    def __calc_gram_matrix(self, X, kenrel_func=0):
        """
        Calculate the Gram matrix, X^T * X, for wolfe dual problem
        """
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.__linear__kernel_func(X[i], X[j])
        return K

    def fit(self, X, y):
        n_samples, n_features = X.shape

        K = self.__calc_gram_matrix(X)

        # Solve the dual problem
         # Set up the quadratic programming problem
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones((n_samples, 1)))
        G = matrix(-np.eye(n_samples))
        h = matrix(np.zeros(n_samples))
        A = matrix(y.reshape(1, -1).astype(np.double))
        b = matrix(np.zeros(1))

        # Solve the quadratic programming problem
        solution = solvers.qp(P, q, G, h, A, b)

        # Extract Lagrange multipliers
        lagrange_multipliers = np.array(solution['x']).flatten()

        # Find support vectors
        sv_threshold = 1e-5
        sv = lagrange_multipliers > sv_threshold
        self.lagrange_multipliers = lagrange_multipliers[sv]
        self.support_vectors = X[sv]
        self.support_vector_labels = y[sv]

        # Compute w and b
        self.w = np.sum(self.lagrange_multipliers.reshape(-1,1) * 
                        self.support_vector_labels.reshape(-1,1) * 
                        self.support_vectors, axis=0)
        self.w0 = np.mean(self.support_vector_labels - np.dot(self.support_vectors, self.w))

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.w0)

def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.r_[np.random.randn(n_samples, 2) - [2, 2], np.random.randn(n_samples, 2) + [2, 2]]
    y = np.array([1] * n_samples + [-1] * n_samples)
    return X, y

import matplotlib.pyplot as plt
def plot_svm(X, y, svm):
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')

    # Plot the decision boundary
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # Plot the support vectors
    plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k')

    plt.title('SVM Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Generate data and train SVM
X, y = generate_data(n_samples=100)
svm = SVM(C=1.0)
svm.fit(X, y)

# Plot the results
plot_svm(X, y, svm)

print("Number of support vectors:", len(svm.support_vectors))
print("w:", svm.w)
print("w0:", svm.w0)