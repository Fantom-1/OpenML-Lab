# 📉 Linear Models Module

The `linear_model` module implements fundamental mapping algorithms that serve as the bedrock of machine learning. We focus on Ordinary Least Squares (OLS) and Regularized variations to manage the **Bias-Variance Tradeoff**.

## 1. Ordinary Least Squares (OLS)
The objective is to minimize the Residual Sum of Squares (RSS):
$$J(\theta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Solvers:
* **Normal Equation:** An analytical solution for global minima.
    $$\hat{\theta} = (X^T X)^{-1} X^T y$$
* **Stochastic Gradient Descent (SGD):** An iterative optimizer for large-scale datasets.
    $$\theta := \theta - \alpha \nabla_{\theta} J(\theta)$$

---

## 2. Regularization (The Armor of the Model)
When features are highly correlated (**Multicollinearity**) or the model is over-complex, we add a penalty term $\alpha R(w)$ to the loss function.

### A. Ridge Regression (L2 Regularization)
Ridge adds a penalty equal to the square of the magnitude of coefficients.
$$J(w) = \text{MSE} + \alpha \sum_{j=1}^{m} w_j^2$$
* **Characteristics:** Shrinks weights asymptotically toward zero. 
* **Use Case:** Ideal for handling multicollinearity and ensuring model stability. It keeps all features but minimizes their individual impact.

### B. Lasso Regression (L1 Regularization)
Lasso adds a penalty equal to the absolute value of the magnitude of coefficients.
$$J(w) = \text{MSE} + \alpha \sum_{j=1}^{m} |w_j|$$
* **Characteristics:** Induces **Sparsity**. Due to the diamond-shaped geometry of the L1 constraint, it forces less important weights to become **exactly zero**.
* **Use Case:** Automatic **Feature Selection**. Use this when you suspect only a small subset of features actually drives the output.

---

## 🏗️ Implementation Guidelines
1.  **Scaling:** Always use `StandardScaler` before training `SGDRegressor` or `LassoRegression`. Regularization is sensitive to feature scale.
2.  **Bias Exclusion:** We do not regularize the intercept ($w_0$). Doing so would force the model toward the origin and bias our predictions.