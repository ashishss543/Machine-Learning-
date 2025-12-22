from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

print("Loading dataset...")

# Load dataset
data = load_iris()
X = data.data
y = data.target

print("Splitting data...")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Training Naive Bayes model...")

# Create and train model
model = GaussianNB()
model.fit(X_train, y_train)

print("Predicting test data...")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- RESULTS ---")
print("Confusion Matrix:")
print(cm)

print("\nAccuracy:")
print(accuracy)
