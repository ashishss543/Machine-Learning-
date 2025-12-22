# future_sales_prediction.py

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Sample Sales Dataset
# -----------------------------
# Advertising spend (in lakhs)
X = np.array([10, 15, 20, 25, 30, 35, 40, 45]).reshape(-1, 1)

# Corresponding sales (in lakhs)
y = np.array([25, 30, 35, 45, 50, 55, 60, 65])

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# -----------------------------
# Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
print("Actual Sales :", y_test)
print("Predicted Sales :", y_pred)

print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -----------------------------
# Future Sales Prediction
# -----------------------------
future_ad_spend = [[50]]  # 50 lakhs advertising
future_sales = model.predict(future_ad_spend)

print("\nPredicted Future Sales for 50 Lakhs Advertising:")
print("Sales =", future_sales[0], "Lakhs")
