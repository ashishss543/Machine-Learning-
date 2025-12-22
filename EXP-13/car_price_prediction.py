# car_price_prediction.py

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------------------
# Sample Car Dataset
# ---------------------------
# Features: [Age (years), Mileage (km), EngineSize (cc)]
X = [
    [1, 15000, 1200],
    [2, 30000, 1500],
    [3, 45000, 1500],
    [4, 60000, 1800],
    [5, 75000, 2000],
    [6, 90000, 2000],
    [7, 105000, 2200],
    [8, 120000, 2500]
]

# Prices (in Lakhs)
y = [9.5, 8.7, 8.0, 7.2, 6.5, 6.0, 5.5, 5.0]

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# ---------------------------
# Linear Regression Model
# ---------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------
# Prediction
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# Results
# ---------------------------
print("Predicted Prices:", y_pred)
print("Actual Prices   :", y_test)

print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))

# ---------------------------
# New Car Price Prediction
# ---------------------------
new_car = [[3, 40000, 1500]]  # Age=3 yrs, Mileage=40k km, Engine=1500cc
predicted_price = model.predict(new_car)

print("\nPredicted Price for New Car:")
print("₹", round(predicted_price[0], 2), "Lakhs")
