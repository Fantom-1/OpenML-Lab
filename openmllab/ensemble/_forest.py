from ..tree import DecisionTreeClassifier
import numpy as np

class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split
            )
            # 1. Create a bootstrap sample of X and y
            # 2. Fit the tree
            # 3. Add to self.trees
            pass

    def predict(self, X):
        # 1. Get predictions from every tree
        # 2. Perform a "Majority Vote"
        pass