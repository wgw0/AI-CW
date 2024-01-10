import pickle
import numpy as np
import matplotlib.pyplot as plt

# Function to load CIFAR-10 dataset
def load_cifar10():
    # Open the CIFAR-10 data file in binary read ('rb') mode
    with open('cifar-10/data_batch_1', 'rb') as fo:
        # Load the data from the file using pickle and decode it using 'bytes' encoding
        data = pickle.load(fo, encoding='bytes')
    # Return the loaded data
    return data

# Load CIFAR-10 dataset by calling the load_cifar10() function
cifar_data = load_cifar10()

# Choose a random image index from the loaded dataset
random_index = np.random.randint(0, cifar_data[b'data'].shape[0])

# Get the image and its corresponding label using the random index
image = cifar_data[b'data'][random_index].reshape(3, 32, 32).transpose(1, 2, 0)
label = cifar_data[b'labels'][random_index]

# Define CIFAR-10 class names for label interpretation
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Create a new figure for displaying the image
plt.figure(figsize=(8, 8))

# Plotting the R, G, B channels separately
for i in range(3):
    # Create a subplot for each color channel (R, G, B)
    plt.subplot(1, 3, i + 1)
    # Display the individual color channel as an image using imshow()
    plt.imshow(image[:,:,i], cmap='gray')
    # Set the title for the subplot
    plt.title(f"Channel {i + 1}")
    # Turn off axis labels for cleaner visualisation
    plt.axis('off')

# Set a super title for the entire figure, indicating the label of the image
plt.suptitle(f"Label: {class_names[label]}", y=0.92)

# Display the figure with the image and channel subplots
plt.show()