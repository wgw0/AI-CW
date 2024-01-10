import numpy as np

# This code does not work, it does not provide a correct output.

# ReLU function for activation
def relu(x):
    return np.maximum(0, x)

# Derivative of the ReLU function for backpropagation
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

class SingleLayerNN(object):
    def __init__(self, input_size, output_size):
        # Initialise weights randomly with mean 0
        self.synaptic_weights = 2 * np.random.random((input_size, output_size)) - 1

    def train(self, training_inputs, training_outputs, iterations):
        for iteration in range(iterations):
            # Forward propagation
            outputs = self.forward(training_inputs)

            # Back propagation
            error = training_outputs - outputs
            adjustments = error * relu_derivative(outputs)
            self.synaptic_weights += np.dot(training_inputs.T, adjustments)

    def forward(self, inputs):
        # Forward propagation through our network
        return relu(np.dot(inputs, self.synaptic_weights))

# Create training and target data
training_inputs = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
training_outputs = np.array([[0,1,1,0]]).T

# Create a single layer neural network
nn = SingleLayerNN(input_size=3, output_size=1)

# Train the neural network
nn.train(training_inputs, training_outputs, 10000)

# Test data
test_inputs = np.array([[1,0,1], [0,0,0], [1,0,0]])
test_outputs = nn.forward(test_inputs)

# Display results
print("Test Inputs:")
print(test_inputs)
print("Test Outputs (Predictions):")
print(test_outputs)
