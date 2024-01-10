import numpy as np
from sklearn.linear_model import LogisticRegression
from keras.datasets import cifar10
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Flatten the images
train_images_flattened = train_images.reshape(train_images.shape[0], -1)
test_images_flattened = test_images.reshape(test_images.shape[0], -1)

# Convert labels to 1D array
train_labels = train_labels.ravel()
test_labels = test_labels.ravel()

# Initialize a Logistic Regression model (a simple linear classifier)
model = LogisticRegression(max_iter=100, solver='saga', multi_class='multinomial')

# Train the model on a subset of the training data for demonstration
model.fit(train_images_flattened[:1000], train_labels[:1000])

# Access the learned templates (weights)
templates = model.coef_

# Reshape and visualize the templates
# Each class has a template, reshape them to the original image size
num_classes = 10
fig, axes = plt.subplots(1, num_classes, figsize=(20, 2))
for i in range(num_classes):
    ax = axes[i]
    template = templates[i].reshape(32, 32, 3)
    ax.imshow(template)
    ax.axis('off')
    ax.set_title(f'Class {i}')

plt.show()