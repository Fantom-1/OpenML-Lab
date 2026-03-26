import numpy as np
from openmllab.tree import DecisionTreeClassifier

# XOR Problem (Non-Linear)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X, y)

preds = tree.predict(X)
print(f"XOR Truth: {y}")
print(f"XOR Preds: {preds}")