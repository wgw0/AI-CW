import pickle
import numpy as np
import matplotlib.pyplot as plt

# Constants
CIFAR10_DATA_FILE = 'cifar-10/data_batch_1'  # File path for CIFAR-10 dataset
ENCODING_TYPE = 'bytes'  # Encoding type for reading the data
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR-10 class names

def load_cifar10(filepath):
    # Load CIFAR-10 dataset from the specified file
    with open(filepath, 'rb') as file:
        data = pickle.load(file, encoding=ENCODING_TYPE)  # Loading data using pickle
    return data

def select_random_image(data):
    # Select a random image and its label from the dataset
    index = np.random.randint(0, data[b'data'].shape[0])  # Random index for selecting an image
    image = data[b'data'][index].reshape(3, 32, 32).transpose(1, 2, 0)  # Reshape and transpose the image for display
    label = data[b'labels'][index]  # Retrieve the corresponding label
    return image, label

def display_image_and_channels(image, label):
    # Display an image and its R, G, B channels separately
    plt.figure(figsize=(8, 8))  # Creating a figure for display
    for i in range(3):
        plt.subplot(1, 3, i + 1)  # Create a subplot for each channel
        plt.imshow(image[:, :, i], cmap='gray')  # Displaying each channel
        plt.title(f"Channel {i + 1}")  # Title for each channel
        plt.axis('off')  # Hide axis for a cleaner look

    plt.suptitle(f"Label: {CLASS_NAMES[label]}", y=0.92)  # Super title for the figure
    plt.show()  # Display the figure

# Main execution
if __name__ == "__main__":
    cifar_data = load_cifar10(CIFAR10_DATA_FILE)  # Load CIFAR-10 data
    image, label = select_random_image(cifar_data)  # Select a random image and its label
    display_image_and_channels(image, label)  # Display the selected image and channels
