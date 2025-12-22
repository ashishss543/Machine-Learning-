# File Name: iris_naive_bayes.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------
# Load Iris Dataset
# ---------------------------
iris = load_iris()
X = iris.data
y = iris.target

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# ---------------------------
# Naive Bayes Model
# ---------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# ---------------------------
# Prediction
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# Results
# ---------------------------
print("Predicted Values:")
print(y_pred)

print("\nActual Values:")
print(y_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------
# New Sample Classification
# ---------------------------
new_sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_sample)

print("\nNew Sample Prediction:")
print("Flower Class:", iris.target_names[prediction[0]])
