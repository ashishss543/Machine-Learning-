from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Load Iris Dataset
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# -----------------------------
# KNN Model
# -----------------------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = knn.predict(X_test)

# -----------------------------
# Results
# -----------------------------
print("Predicted Labels:", y_pred)
print("Actual Labels   :", y_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Classify New Flower
# -----------------------------
# Example: [SepalLength, SepalWidth, PetalLength, PetalWidth]
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = knn.predict(new_flower)

print("\nNew Flower Classification:")
print("Flower Type:", iris.target_names[prediction[0]])
