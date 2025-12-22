import math
from collections import Counter

# Euclidean distance function
def euclidean_distance(p1, p2):
    distance = 0
    for i in range(len(p1)):
        distance += (p1[i] - p2[i]) ** 2
    return math.sqrt(distance)

# KNN function
def knn(training_data, training_labels, test_point, k):
    distances = []

    for i in range(len(training_data)):
        dist = euclidean_distance(training_data[i], test_point)
        distances.append((dist, training_labels[i]))

    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    class_votes = [label for _, label in neighbors]
    return Counter(class_votes).most_common(1)[0][0]

# Training dataset
X_train = [
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 5],
    [7, 7],
    [8, 6]
]

y_train = ['A', 'A', 'A', 'B', 'B', 'B']

# Test data
test_sample = [5, 5]
k = 3

# Prediction
result = knn(X_train, y_train, test_sample, k)
print("Predicted Class:", result)
