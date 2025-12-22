# Perceptron based IRIS Classification
# Lab Experiment Code

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Load IRIS Dataset
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# -----------------------------
# Perceptron Model
# -----------------------------
model = Perceptron(max_iter=1000, eta0=0.1, random_state=1)
model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Results
# -----------------------------
print("Predicted Labels:", y_pred)
print("Actual Labels   :", y_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
