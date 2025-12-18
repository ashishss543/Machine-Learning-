import math
from collections import Counter

# Dataset
data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]

attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']

# Entropy calculation
def entropy(data):
    labels = [row[-1] for row in data]
    counts = Counter(labels)
    ent = 0
    for count in counts.values():
        p = count / len(labels)
        ent -= p * math.log2(p)
    return ent

# Information Gain
def information_gain(data, attr_index):
    total_entropy = entropy(data)
    values = set(row[attr_index] for row in data)
    weighted_entropy = 0

    for value in values:
        subset = [row for row in data if row[attr_index] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)

    return total_entropy - weighted_entropy

# ID3 Algorithm
def id3(data, attrs):
    labels = [row[-1] for row in data]

    # If all labels same, return label
    if labels.count(labels[0]) == len(labels):
        return labels[0]

    # If no attributes left
    if not attrs:
        return Counter(labels).most_common(1)[0][0]

    # Best attribute selection
    gains = [information_gain(data, i) for i in range(len(attrs))]
    best_attr = gains.index(max(gains))
    tree = {attrs[best_attr]: {}}

    attr_values = set(row[best_attr] for row in data)

    for value in attr_values:
        subset = [row[:best_attr] + row[best_attr+1:] for row in data if row[best_attr] == value]
        sub_attrs = attrs[:best_attr] + attrs[best_attr+1:]
        tree[attrs[best_attr]][value] = id3(subset, sub_attrs)

    return tree

# Build decision tree
decision_tree = id3(data, attributes)

print("Decision Tree:")
print(decision_tree)

# Classify new sample
def classify(tree, attrs, sample):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    index = attrs.index(attr)
    value = sample[index]
    return classify(tree[attr][value], attrs, sample)

# New sample
new_sample = ['Sunny', 'Cool', 'High', 'Strong']
result = classify(decision_tree, attributes, new_sample)

print("\nNew Sample:", new_sample)
print("Prediction:", result)
