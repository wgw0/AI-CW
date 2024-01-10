import numpy as np

# ReLU function for activation
def relu(x):
    return np.maximum(0, x)

# Derivative of the ReLU function for backpropagation
def relu_derivative(x):
    return (x > 0).astype(float)

# Define a Two Layer Neural Network class
class TwoLayerNN(object):
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.learning_rate = learning_rate
        self.synaptic_weights1 = 2 * np.random.random((input_size, hidden_size)) - 1
        self.synaptic_weights2 = 2 * np.random.random((hidden_size, output_size)) - 1

    def train(self, training_inputs, training_outputs, iterations):
        for iteration in range(iterations):
            # Forward propagation
            output_from_layer1, output_from_layer2 = self.forward(training_inputs)

            # Back propagation
            layer2_error = training_outputs - output_from_layer2
            layer2_adjustments = layer2_error * relu_derivative(output_from_layer2)

            layer1_error = layer2_adjustments.dot(self.synaptic_weights2.T)
            layer1_adjustments = layer1_error * relu_derivative(output_from_layer1)

            # Update weights
            self.synaptic_weights1 += self.learning_rate * training_inputs.T.dot(layer1_adjustments)
            self.synaptic_weights2 += self.learning_rate * output_from_layer1.T.dot(layer2_adjustments)

    def forward(self, inputs):
        # Forward propagation through the network
        output_from_layer1 = relu(np.dot(inputs, self.synaptic_weights1))
        output_from_layer2 = relu(np.dot(output_from_layer1, self.synaptic_weights2))
        return output_from_layer1, output_from_layer2

# Create training and target data
training_inputs = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
training_outputs = np.array([[0,1,1,0]]).T

# Create a two layer neural network with a learning rate
learning_rate = 0.1
nn = TwoLayerNN(input_size=3, hidden_size=4, output_size=1, learning_rate=learning_rate)

# Train the neural network
nn.train(training_inputs, training_outputs, 10000)

# Test data
test_inputs = np.array([[1,1,0], [0,0,0], [1,0,0]])
_, test_outputs = nn.forward(test_inputs)

# Display results
print("Test Inputs:")
print(test_inputs)
print("Test Outputs (Predictions):")
print(test_outputs)
