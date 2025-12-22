# File Name: mobile_price_prediction.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Sample Mobile Dataset
# Features: [RAM, Battery, Storage, Camera]
# Target: Price Range (0=Low, 1=Medium, 2=High)
# -----------------------------
X = [
    [2000, 3000, 32, 8],
    [4000, 4000, 64, 12],
    [6000, 5000, 128, 48],
    [3000, 3500, 32, 13],
    [8000, 6000, 256, 64],
    [1000, 2500, 16, 5],
    [5000, 4500, 128, 48],
    [7000, 5500, 256, 50]
]

y = [0, 1, 2, 1, 2, 0, 2, 2]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# Model Training
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# Prediction & Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("Predicted Values:", y_pred)
print("Actual Values   :", y_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# New Mobile Prediction
# -----------------------------
new_mobile = [[6000, 4800, 128, 48]]
result = model.predict(new_mobile)

print("\nNew Mobile Price Category:")
if result[0] == 0:
    print("Low Price")
elif result[0] == 1:
    print("Medium Price")
else:
    print("High Price")
