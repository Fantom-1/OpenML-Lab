import numpy as np
from ..linear_model._base import BaseEstimator

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier(BaseEstimator):
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        values, counts = np.unique(y, return_counts=True)
        return 1 - np.sum((counts / m) ** 2)

    def _best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_gini = float("inf")

        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                y_left, y_right = y[left_mask], y[right_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini = (
                    len(y_left) / n_samples * self._gini(y_left) +
                    len(y_right) / n_samples * self._gini(y_right)
                )

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping conditions
        if (
            depth >= self.max_depth or
            n_labels == 1 or
            n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find best split
        feature, threshold = self._best_split(X, y)

        if feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature, threshold, left_child, right_child)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
        
    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])