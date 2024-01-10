import numpy as np

# Sigmoid function for activation
# This function maps any input value to a value between 0 and 1.
# It's used as the activation function for the neural network.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function for backpropagation
# This function calculates the gradient of the sigmoid curve.
# It's used to adjust weights during backpropagation.
def sigmoid_derivative(x):
    return x * (1 - x)

# Define a Single Layer Neural Network class
class SingleLayerNN(object):
    def __init__(self, input_size, output_size):
        # Initialise weights randomly with mean 0
        # The weights link the input layer to the output layer directly in this single-layer network.
        self.synaptic_weights = 2 * np.random.random((input_size, output_size)) - 1

    def train(self, training_inputs, training_outputs, iterations):
        for iteration in range(iterations):
            # Forward propagation:
            # Pass the training set through our neural network to get the output.
            outputs = self.forward(training_inputs)

            # Back propagation:
            # Calculate the error, which is the difference between desired output and predicted output.
            error = training_outputs - outputs

            # Calculate adjustments based on the error and the derivative of the sigmoid function.
            # This helps in minimising the error in predictions during the next iterations.
            adjustments = error * sigmoid_derivative(outputs)

            # Update the weights for better predictions in the next iteration.
            self.synaptic_weights += np.dot(training_inputs.T, adjustments)

    def forward(self, inputs):
        # Forward propagation through our network
        # This function takes the inputs and passes them through the network to get the output.
        return sigmoid(np.dot(inputs, self.synaptic_weights))

# Create training and target data
# The training set consists of examples with known inputs and outputs.
training_inputs = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
training_outputs = np.array([[0,1,1,0]]).T

# Create a single layer neural network
nn = SingleLayerNN(input_size=3, output_size=1)

# Train the neural network
# This process adjusts the weights of the network based on the training data.
nn.train(training_inputs, training_outputs, 10000)

# Test data
# New data to test the performance of the neural network after training.
test_inputs = np.array([[1,1,0], [0,0,0], [1,0,0]])
test_outputs = nn.forward(test_inputs)

# Display results
# Print the test data and the corresponding predictions from the network.
print("Test Inputs:")
print(test_inputs)
print("Target Outputs")
print()
print("Test Outputs (Predictions):")
print(test_outputs)
