import numpy as np

# Training data for OR and XOR operations
or_data = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}
xor_data = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}

# Activation function: Unit Step Function
def unit_step(v):
    return 1 if v >= 0 else 0

# Perceptron Function
def perceptron(theta, x, w, b):
    return theta(np.dot(w, x) + b)

# OR Perceptron Implementation
def or_perceptron(x):
    return perceptron(unit_step, x, [1, 1], -1)

# XOR Perceptron Attempt
# Implementing a perceptron for XOR with arbitrary weights and bias.
# Note: There's no set of weights and bias that will make this work for XOR as it's not linearly separable.
def xor_perceptron(x):
    return perceptron(unit_step, x, [1, 1], -0.5)

# Testing the OR Perceptron
print("Testing OR Perceptron")
for input in or_data:
    print(f"Input: {input}, Perceptron Answer: {or_perceptron(input)}, Expected: {or_data[input]}")

# Testing the XOR Perceptron
# This will demonstrate that a single-layer perceptron cannot correctly solve the XOR problem.
print("\nTesting XOR Perceptron")
for input in xor_data:
    result = xor_perceptron(input)
    print(f"Input: {input}, Perceptron Answer: {result}, Expected: {xor_data[input]}")
    if result != xor_data[input]:
        print("  ^ Incorrect result - Demonstrates failure to solve XOR with a single-layer perceptron")