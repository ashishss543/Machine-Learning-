import numpy as np

# Sample data (1D)
X = np.array([1.0, 1.2, 1.8, 5.0, 5.5, 6.0])

# Number of clusters
k = 2
n = len(X)

# Initialize parameters
means = np.array([1.0, 5.0])
variances = np.array([1.0, 1.0])
weights = np.array([0.5, 0.5])

# Gaussian function
def gaussian(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean)**2 / (2 * var))

# EM Algorithm
for iteration in range(5):
    print("\nIteration", iteration + 1)

    # E-step
    responsibilities = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            responsibilities[i][j] = weights[j] * gaussian(X[i], means[j], variances[j])
        responsibilities[i] /= np.sum(responsibilities[i])

    # M-step
    for j in range(k):
        r = responsibilities[:, j]
        total_r = np.sum(r)

        means[j] = np.sum(r * X) / total_r
        variances[j] = np.sum(r * (X - means[j])**2) / total_r
        weights[j] = total_r / n

    print("Means:", means)
    print("Variances:", variances)
    print("Weights:", weights)

# Final cluster assignment
clusters = np.argmax(responsibilities, axis=1)
print("\nFinal Cluster Assignments:")
for i in range(n):
    print(f"Data point {X[i]} → Cluster {clusters[i]}")
