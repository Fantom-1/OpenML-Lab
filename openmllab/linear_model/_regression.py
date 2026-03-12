import numpy as np
from ._base import BaseEstimator

class LinearRegression(BaseEstimator):
    """
    Ordinary Least Squares using the Normal Equation.
    Best for small to medium datasets.
    """
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        # The Bias Trick: Adding a column of 1s
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Normal Equation: theta = (X^T @ X)^-1 @ X^T @ y
        # We use pinv (pseudo-inverse) for numerical stability
        self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        X = np.asarray(X)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta


class SGDRegressor(BaseEstimator):
    """
    Linear Regression using Stochastic Gradient Descent.
    Scales to massive datasets.
    """
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y
            
            # Gradients derived from MSE loss
            dw = (2/n_samples) * np.dot(X.T, error)
            db = (2/n_samples) * np.sum(error)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        X = np.asarray(X)
        return np.dot(X, self.weights) + self.bias