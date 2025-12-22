import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Dataset
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])

# -------- Linear Regression --------
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

# -------- Polynomial Regression (degree = 2) --------
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)

# -------- Evaluation --------
mse_linear = mean_squared_error(y, y_pred_linear)
mse_poly = mean_squared_error(y, y_pred_poly)

# -------- Output --------
print("LINEAR REGRESSION")
print("Predicted values:", y_pred_linear)
print("Mean Squared Error:", mse_linear)

print("\nPOLYNOMIAL REGRESSION (Degree = 2)")
print("Predicted values:", y_pred_poly)
print("Mean Squared Error:", mse_poly)
