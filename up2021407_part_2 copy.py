import numpy as np

# Define the Perceptron class
class Perceptron:
    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)


# Define training data for the OR problem
training_inputs_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels_or = np.array([0, 1, 1, 1])

# Create a perceptron instance for the OR problem
perceptron_or = Perceptron(2)
perceptron_or.train(training_inputs_or, labels_or)

# Testing the OR problem solution
print("Testing OR problem:")
for i in range(len(training_inputs_or)):
    print(f"Input: {training_inputs_or[i]}, Predicted Output: {perceptron_or.predict(training_inputs_or[i])}")

# Define training data for the XOR problem
training_inputs_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels_xor = np.array([0, 1, 1, 0])

# Create a perceptron instance for the XOR problem
perceptron_xor = Perceptron(2)
perceptron_xor.train(training_inputs_xor, labels_xor)

# Testing the inability to solve the XOR problem
print("\nTesting XOR problem:")
for i in range(len(training_inputs_xor)):
    print(f"Input: {training_inputs_xor[i]}, Predicted Output: {perceptron_xor.predict(training_inputs_xor[i])}")

