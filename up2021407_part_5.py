import pickle
import numpy as np
import matplotlib.pyplot as plt

# Function to load CIFAR-10 dataset
def load_cifar10():
    with open('cifar-10/data_batch_1', 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

# Load CIFAR-10 dataset
cifar_data = load_cifar10()

# Choose a random image index
random_index = np.random.randint(0, cifar_data[b'data'].shape[0])

# Get the image and its corresponding label
image = cifar_data[b'data'][random_index].reshape(3, 32, 32).transpose(1, 2, 0)
label = cifar_data[b'labels'][random_index]

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Display the image as a graph
plt.figure(figsize=(8, 8))

# Plotting the R, G, B channels separately
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(image[:,:,i], cmap='gray')
    plt.title(f"Channel {i + 1}")
    plt.axis('off')

plt.suptitle(f"Label: {class_names[label]}", y=0.92)
plt.show()