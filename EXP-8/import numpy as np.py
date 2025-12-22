import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Training data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([0, 0, 0, 1, 1])

# Initialize weights and bias
w = 0
b = 0
lr = 0.1
epochs = 1000

# Training using Gradient Descent
for _ in range(epochs):
    for i in range(len(X)):
        z = w * X[i] + b
        y_pred = sigmoid(z)
        error = Y[i] - y_pred
        w += lr * error * X[i]
        b += lr * error

# Test new value
x_test = 3.5
z = w * x_test + b
prediction = sigmoid(z)

print("Weight:", w)
print("Bias:", b)
print("Probability:", prediction)

if prediction >= 0.5:
    print("Class: 1 (Positive)")
else:
    print("Class: 0 (Negative)")
