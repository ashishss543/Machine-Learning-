import random
import math

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Training data (XOR)
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

outputs = [0, 1, 1, 0]

# Initialize weights
w1 = random.random()
w2 = random.random()
w3 = random.random()
w4 = random.random()
w5 = random.random()
w6 = random.random()

learning_rate = 0.5

# Training
for epoch in range(10000):
    for i in range(len(inputs)):
        x1, x2 = inputs[i]
        target = outputs[i]

        # Forward propagation
        h1 = sigmoid(x1 * w1 + x2 * w2)
        h2 = sigmoid(x1 * w3 + x2 * w4)
        output = sigmoid(h1 * w5 + h2 * w6)

        # Error
        error = target - output

        # Backpropagation
        d_output = error * sigmoid_derivative(output)

        d_w5 = d_output * h1
        d_w6 = d_output * h2

        d_h1 = d_output * w5 * sigmoid_derivative(h1)
        d_h2 = d_output * w6 * sigmoid_derivative(h2)

        d_w1 = d_h1 * x1
        d_w2 = d_h1 * x2
        d_w3 = d_h2 * x1
        d_w4 = d_h2 * x2

        # Update weights
        w1 += learning_rate * d_w1
        w2 += learning_rate * d_w2
        w3 += learning_rate * d_w3
        w4 += learning_rate * d_w4
        w5 += learning_rate * d_w5
        w6 += learning_rate * d_w6

# Testing
print("Testing the trained ANN:")
for i in range(len(inputs)):
    x1, x2 = inputs[i]
    h1 = sigmoid(x1 * w1 + x2 * w2)
    h2 = sigmoid(x1 * w3 + x2 * w4)
    output = sigmoid(h1 * w5 + h2 * w6)
    print(f"Input: {inputs[i]} Output: {round(output)}")
