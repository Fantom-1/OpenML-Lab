# 📊 Metrics Module

This module provides the "Judicial System" for OpenMLLab. We implement evaluation metrics from first principles to verify model performance across different data distributions.

## 1. Regression Metrics
Regression is about minimizing the residual $e_i = y_i - \hat{y}_i$.

### Mean Squared Error (MSE)
Used to punish large outliers by squaring the error term:
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Mean Absolute Error (MAE)
Provides a linear scale of error, useful when outliers should not dominate the metric:
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

---

## 2. Classification Metrics
Classification is evaluated via the **Confusion Matrix**:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### Precision vs. Recall
* **Precision**: The accuracy of positive predictions.
  $$\text{Precision} = \frac{TP}{TP + FP}$$
* **Recall**: The ability to find all positive instances.
  $$\text{Recall} = \frac{TP}{TP + FN}$$

### F1-Score
The harmonic mean of precision and recall, providing a single score for imbalanced datasets:
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
