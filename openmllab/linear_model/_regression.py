import numpy as np
from ._base import BaseEstimator

class LinearRegression(BaseEstimator):
    """
    Ordinary Least Squares Linear Regression.
    Solves using the Normal Equation: (X^T * X)^-1 * X^T * y
    """
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Add a column of ones to X for the bias term (intercept)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # The Normal Equation
        # We use np.linalg.inv to find the inverse. 
        # Note: If X^T @ X is singular (not invertible), this will crash.
        # A legend would use np.linalg.pinv (pseudo-inverse) for robustness.
        self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        X = np.asarray(X)
        # Add the same column of ones to the input X
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta