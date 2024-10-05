import sys
import numpy as np
from libtiff import TIFF
from matplotlib import pyplot as plt

# Function 1: Load the grayscale image
def load_image(image_path):
    """
    Load a grayscale image from the given image path.
    
    Args:
        image_path (str): Path to the grayscale image.

    Returns:
        np.ndarray: Loaded image as a numpy array.
    """
    tifImg = TIFF.open(image_path)
    image = TIFF.read_image(tifImg)
    tifImg.close()
    return image

# Function 2: Compute histogram of the grayscale image
def compute_histogram(image):
    """
    Compute the histogram of a grayscale image.
    
    Args:
        image (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: Histogram array with 256 bins, one for each intensity value.
    """
    # Flatten the image to a 1D array and use np.bincount to count pixel intensities
    histogram = np.bincount(image.ravel(), minlength=256)
    
    return histogram

# Function 3: Plot the histogram
def plot_histogram(histogram):
    """
    Plot the histogram of grayscale intensities.
    
    Args:
        histogram (np.ndarray): Histogram array with 256 bins.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(range(256), histogram, color='gray')
    plt.title("Grayscale Intensity Histogram")
    plt.xlabel("Intensity Value")
    plt.ylabel("Pixel Count")
    plt.show()

# Main function to load image, compute histogram, and plot it
def main(image_path):
    # Load the image
    image = load_image(image_path)
    
    # Compute the histogram
    histogram = compute_histogram(image)
    
    # Plot the histogram
    plot_histogram(histogram)

# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print("Usage: python script.py <image_path>")
    sys.exit(1)

# Run the main function with the image path from command-line argument
image_path = sys.argv[1]
main(image_path)
