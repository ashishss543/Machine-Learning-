from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------
# Sample Credit Data
# ---------------------------
# Features: [Income, Age, LoanAmount]
X = [
    [50000, 25, 20000],
    [60000, 45, 15000],
    [30000, 22, 25000],
    [80000, 35, 10000],
    [20000, 21, 30000],
    [90000, 50, 5000],
    [40000, 28, 22000],
    [70000, 40, 12000]
]

# Labels: 1 = Good Credit, 0 = Bad Credit
y = [1, 1, 0, 1, 0, 1, 0, 1]

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# ---------------------------
# Logistic Regression Model
# ---------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------------------
# Prediction
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# Results
# ---------------------------
print("Predicted Values:", y_pred)
print("Actual Values   :", y_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------
# New Customer Prediction
# ---------------------------
new_customer = [[65000, 30, 18000]]
prediction = model.predict(new_customer)

print("\nNew Customer Credit Score:")
if prediction[0] == 1:
    print("Good Credit")
else:
    print("Bad Credit")
