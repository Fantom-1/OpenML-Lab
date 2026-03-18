import numpy as np
from ._base import BaseEstimator

class LogisticRegression(BaseEstimator):
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1/(1+ np.exp(-z))

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        n_samples, n_features = X.shape
        self.weights = np.ones(n_features)

        for _ in range(self.epochs):
            y_prob = np.dot(X, self.weights) + self.bias
            y_hat = self._sigmoid(y_prob)

            error = y_hat - y

            dw = (1/n_samples)*np.dot(X.T, error)
            db = (1/n_samples)*np.sum(error)

            self.weights -= self.lr*dw
            self.bias -=self.lr*db

    def predict_proba(self, X):
       X = np.asarray(X)
       Z = np.dot(X, self.weights) + self.bias
       return self._sigmoid(Z)
    
    def predict(self, X):
        X = np.asarray(X)
        return [1 if p > 0.5 else 0 for p in self.predict_proba(X)]

   