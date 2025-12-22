# bank_loan_naive_bayes.py

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Sample Bank Loan Dataset
# Features: [Income, CreditScore, LoanAmount]
# -----------------------------
X = [
    [50000, 700, 200000],
    [30000, 600, 150000],
    [80000, 750, 300000],
    [20000, 580, 100000],
    [90000, 800, 350000],
    [40000, 620, 180000],
    [100000, 820, 400000],
    [25000, 590, 120000]
]

# Labels: 1 = Loan Approved, 0 = Loan Rejected
y = [1, 0, 1, 0, 1, 0, 1, 0]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)

# -----------------------------
# Naive Bayes Model
# -----------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Results
# -----------------------------
print("Predicted Values:", y_pred)
print("Actual Values   :", y_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# New Customer Prediction
# -----------------------------
new_customer = [[60000, 710, 220000]]
result = model.predict(new_customer)

print("\nNew Customer Loan Status:")
if result[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")
