import numpy as np
from openmllab.linear_model import LinearRegression, SGDRegressor
from openmllab.preprocessing import StandardScaler
from openmllab.metrics import mean_squared_error, r2_score # Note: We need R2 score!

# 1. Generate Synthetic Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1.2, 1.9, 3.0, 4.1, 5.2])

# 2. Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train Model
model = LinearRegression()
model.fit(X_scaled, y)

# 4. Predict and Evaluate
predictions = model.predict(X_scaled)
mse = mean_squared_error(y, predictions)

print(f"Predictions: {predictions}")
print(f"MSE: {mse:.4f}")