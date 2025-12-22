# house_price_prediction.py

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -----------------------------
# Sample House Price Dataset
# -----------------------------
# Features: [Area (sqft), Bedrooms, Age of house]
X = [
    [1000, 2, 10],
    [1500, 3, 5],
    [800, 2, 20],
    [2000, 4, 2],
    [1200, 3, 15],
    [1800, 4, 8],
    [900, 2, 18],
    [1600, 3, 6]
]

# Target: House Price (in lakhs)
y = [50, 75, 40, 100, 60, 90, 45, 80]

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
# Prediction on Test Data
# -----------------------------
y_pred = model.predict(X_test)

print("Predicted Prices:", y_pred)
print("Actual Prices   :", y_test)

# -----------------------------
# Error Calculation
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

# -----------------------------
# New House Prediction
# -----------------------------
# Example: Area=1400 sqft, Bedrooms=3, Age=7 years
new_house = [[1400, 3, 7]]
predicted_price = model.predict(new_house)

print("\nPredicted Price for New House:")
print("₹", round(predicted_price[0], 2), "Lakhs")
