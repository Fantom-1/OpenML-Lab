import numpy as np
from ..linear_model._base import BaseEstimator

class Node:
    """A helper class to represent a node in the decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature     # Index of the feature used for splitting
        self.threshold = threshold # Threshold value for the split
        self.left = left           # Left child (Node)
        self.right = right         # Right child (Node)
        self.value = value         # If it's a leaf node, this stores the class label

class DecisionTreeClassifier(BaseEstimator):
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini(self, y):
   
        m = len(y)
        if m == 0:
         return 0
        else:
         values , counts = np.unique(y,  return_counts=True)
         return 1 - np.sum((counts/m)**2)

         
    

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        # 1. Check stopping criteria (max_depth, min_samples)
        # 2. Find the best split (using Gini or Entropy)
        # 3. Recursively grow left and right subtrees
        pass

    def predict(self, X):
        # Traverse the tree for each sample in X
        pass