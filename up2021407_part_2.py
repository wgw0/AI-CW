# Definition of the OR_perceptron function with two inputs: x1 and x2.
def OR_perceptron(x1, x2):
    # Initializing weights w1 and w2 to 1. These weights are applied to the inputs.
    w1 = 1
    w2 = 1
    # Setting the threshold (theta) value to 1. This is used to decide the output of the perceptron.
    theta = 1
    # Calculating the weighted sum of the inputs.
    yin = (x1 * w1) + (x2 * w2)
    # If the weighted sum is equal or greater than the threshold, return 1 (True).
    if yin >= theta:
        return 1
    # Otherwise, return 0 (False).
    else:
        return 0

# Testing the OR_perceptron function with different sets of inputs.
print("OR Perceptron")
print(OR_perceptron(0, 0))  # Output for inputs (0, 0)
print(OR_perceptron(0, 1))  # Output for inputs (0, 1)
print(OR_perceptron(1, 0))  # Output for inputs (1, 0)
print(OR_perceptron(1, 1))  # Output for inputs (1, 1)
print("----------------------------------")

# Definition of the XOR_perceptron function with two inputs: x1 and x2.
def XOR_perceptron(x1, x2):
    # Initializing weights w1 to 1 and w2 to -1. These weights are applied to the inputs.
    w1 = 1
    w2 = -1
    # Setting the threshold (theta) value to 0. This is used to decide the output of the perceptron.
    theta = 0
    # Calculating the weighted sum of the inputs.
    yin = (x1 * w1) + (x2 * w2)
    # If the weighted sum is equal or greater than the threshold, return 1 (True).
    if yin >= theta:
        return 1
    # Otherwise, return 0 (False).
    else:
        return 0

# Testing the XOR_perceptron function with different sets of inputs.
print("XOR Perceptron")
print(XOR_perceptron(0, 0))  # Output for inputs (0, 0)
print(XOR_perceptron(0, 1))  # Output for inputs (0, 1)
print(XOR_perceptron(1, 0))  # Output for inputs (1, 0)
print(XOR_perceptron(1, 1))  # Output for inputs (1, 1)
