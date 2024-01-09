import numpy as np
import matplotlib.pyplot as plt

# Data for OR operation with all possible input combinations and their corresponding outputs.
or_data = {
    (0, 0): 0,
    (0, 1): 1,
    (1, 0): 1,
    (1, 1): 1
}

# Data for XOR operation with all possible input combinations and their corresponding outputs.
xor_data = {
    (0, 0): 0,
    (0, 1): 1,
    (1, 0): 1,
    (1, 1): 0
}

# Definition of the unit step function, which is the activation function for the perceptron.
def unit_step(v):
    return 1 if v >= 0 else 0

# General perceptron function that computes the weighted sum and applies the activation function.
def perceptron(theta, x, w, b):
    result = np.dot(w, x) + b  # Weighted sum of inputs and bias
    return theta(result)      # Apply activation function

# Specific implementation of a perceptron for the OR operation.
def or_perceptron(x):
    return perceptron(
        unit_step,           # Activation function
        x,                   # Input vector
        np.array([1, 1]),    # Weights
        -1                   # Bias
    )

# Plotting the OR function
plt.scatter([0, 1, 1], [1, 0, 1], label='One', s=[10, 10, 10])  # Plot points for output 1
plt.scatter([0], [0], label='Zero', s=[10])                      # Plot point for output 0
plt.plot([0, 0.9], [0.9, 0], label='Linear separator')           # Plot linear separator

# Setting layout options for OR plot
plt.xticks(range(2))
plt.yticks(range(2))
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('Output of the Or function over 2 inputs')
plt.legend()
plt.grid(True)

plt.show()  # Display OR plot

# Plotting the XOR function
plt.scatter([0, 1], [1, 0], label='One', s=[10, 10])  # Plot points for output 1
plt.scatter([0, 1], [0, 1], label='Zero', s=[10, 10]) # Plot points for output 0

# Setting layout options for XOR plot
plt.xticks(range(2))
plt.yticks(range(2))
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('Output of the XOR function over 2 inputs')
plt.legend()
plt.grid(True)

plt.show()  # Display XOR plot
