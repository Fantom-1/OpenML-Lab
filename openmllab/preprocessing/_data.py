import numpy as np

class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """Compute the mean and std to be used for later scaling."""
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        """Perform standardization by centering and scaling."""
        X = np.asarray(X)
        # Handle division by zero if std is 0
        scale = np.where(self.scale_ == 0, 1, self.scale_)
        return (X - self.mean_) / scale

    def fit_transform(self, X):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)