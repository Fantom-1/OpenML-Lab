import numpy as np
from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    """Abstract Base Class for all models in OpenMLLab."""

    @abstractmethod
    def fit(self, X, y):
        """Standard method to train the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Standard method to make predictions."""
        pass