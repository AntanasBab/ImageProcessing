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

# Function 2: Compute histogram normalization (equalization)
def compute_histogram_normalization(image):
    """
    Apply histogram normalization to a grayscale image.
    
    Args:
        image (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: Histogram normalized image.
    """
    # Flatten the image to a 1D array
    flat_image = image.ravel()
    
    # Compute the histogram (256 bins for each intensity value)
    histogram, bin_edges = np.histogram(flat_image, bins=256, range=(0, 255))
    
    # Compute the cumulative distribution function (CDF)
    cdf = histogram.cumsum()
    
    # Normalize the CDF to the range [0, 255]
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)  # Convert to uint8

    # Use the CDF to remap the intensity values in the original image
    equalized_image = cdf_normalized[flat_image]
    
    # Reshape back to the original image shape
    equalized_image = equalized_image.reshape(image.shape)
    
    return equalized_image

# Function 3: Display the original and normalized images side by side
def apply_histogram_normalization(image_path):
    # Load the original image
    original_image = load_image(image_path)
    
    # Apply histogram normalization
    normalized_image = compute_histogram_normalization(original_image)
    
    # Plot original and normalized images side by side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    
    plt.subplot(1, 2, 2)
    plt.title("Histogram Normalized Image")
    plt.imshow(normalized_image, cmap='gray', vmin=0, vmax=255)
    
    plt.show()

# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print("Usage: python script.py <image_path>")
    sys.exit(1)

# Run the main function with the image path from command-line argument
image_path = sys.argv[1]
apply_histogram_normalization(image_path)
