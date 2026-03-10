# 📉 Linear Models Module

This module contains implementations of linear mapping algorithms, where the relationship between input $X$ and target $y$ is modeled as a linear combination.

## 1. Ordinary Least Squares (OLS)
The objective of OLS is to find the vector $\hat{\theta}$ that minimizes the Sum of Squared Residuals (SSR):
$$J(\theta) = \sum_{i=1}^{n} (y_i - X_i\theta)^2$$

### The Normal Equation
Instead of iterative optimization, we solve for the global minimum analytically. The derivation follows setting the gradient $\nabla_{\theta} J(\theta) = 0$:

$$\hat{\theta} = (X^T X)^{-1} X^T y$$

**Implementation Details:**
* **The Bias Trick:** We augment $X$ with a column of ones to incorporate the intercept $c$ into the weight vector $\theta$.
* **Numerical Stability:** We utilize the **Moore-Penrose Pseudo-inverse** via SVD to handle rank-deficient (singular) matrices, ensuring the library does not crash on collinear features.

### Complexity Analysis
* **Time Complexity:** $O(n^2 \cdot m + n^3)$ where $n$ is features and $m$ is samples.
* **Space Complexity:** $O(n^2)$ to store the covariance matrix.