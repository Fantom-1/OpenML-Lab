# 🌲 Decision Trees Module

Decision Trees are non-parametric supervised learning methods used for classification. They predict the value of a target variable by learning simple decision rules inferred from the data features.

## 1. Gini Impurity
We measure the quality of a split using the **Gini Impurity**, which quantifies the probability of an incorrectly classified feature:
$$G = 1 - \sum_{i=1}^{c} p_i^2$$
A Gini score of **0** represents a perfectly pure node.

## 2. Recursive Partitioning
Our implementation uses a recursive top-down approach:
1.  **Selection:** Search for the feature and threshold that minimize the weighted Gini impurity.
2.  **Splitting:** Divide the dataset into two subsets (Left/Right) based on the threshold.
3.  **Recursion:** Repeat the process for each subset until a stopping criterion is met.

## 3. Hyperparameters
* `max_depth`: Limits the length of the longest path from root to leaf to prevent **Overfitting**.
* `min_samples_split`: The minimum number of samples required to split an internal node.