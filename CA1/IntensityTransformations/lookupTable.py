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

# Function 2: Create a lookup table
def create_lookup_table(transformation_func):
    """
    Create a lookup table (LUT) for intensity transformation.
    
    Args:
        transformation_func (function): A function that maps an intensity value (0-255) to a new value.

    Returns:
        np.ndarray: Lookup table (LUT) with 256 elements.
    """
    lut = np.array([transformation_func(i) for i in range(256)], dtype=np.uint8)
    return lut

# Function 3: Apply the lookup table to the image
def apply_lut_transformation(image, lut):
    """
    Apply a lookup table (LUT) transformation to an image.
    
    Args:
        image (np.ndarray): Grayscale image (2D array).
        lut (np.ndarray): Lookup table with 256 elements.

    Returns:
        np.ndarray: Transformed image.
    """
    transformed_image = lut[image]
    return transformed_image

# Histogram normalization (equalization) using LUT
def histogram_normalization_lut(image):
    """
    Perform histogram normalization using a lookup table.
    
    Args:
        image (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: Histogram normalized image.
    """
    # Compute the histogram
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 255))
    
    # Compute the cumulative distribution function (CDF)
    cdf = histogram.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())  # Normalize CDF to [0, 255]
    cdf_normalized = cdf_normalized.astype(np.uint8)
    
    # Create a lookup table using the CDF
    lut = create_lookup_table(lambda i: cdf_normalized[i])
    
    # Apply LUT to the image
    return apply_lut_transformation(image, lut)

# Power-law transformation using LUT
def power_law_lut(image, gamma):
    """
    Perform power-law (gamma) transformation using a lookup table.
    
    Args:
        image (np.ndarray): Grayscale image.
        gamma (float): Power-law exponent.

    Returns:
        np.ndarray: Gamma corrected image.
    """
    # Create a lookup table for the power-law transformation
    lut = create_lookup_table(lambda i: np.clip(255 * (i / 255) ** gamma, 0, 255))
    
    # Apply LUT to the image
    return apply_lut_transformation(image, lut)

# Thresholding using LUT
def threshold_lut(image, threshold_value):
    """
    Perform thresholding using a lookup table.
    
    Args:
        image (np.ndarray): Grayscale image.
        threshold_value (int): Threshold value (0-255).

    Returns:
        np.ndarray: Thresholded binary image.
    """
    # Create a lookup table for thresholding
    lut = create_lookup_table(lambda i: 255 if i > threshold_value else 0)
    
    # Apply LUT to the image
    return apply_lut_transformation(image, lut)

# Main function to demonstrate the different LUT transformations
def main(image_path):
    # Load the image
    image = load_image(image_path)
    
    # Apply histogram normalization using LUT
    normalized_image = histogram_normalization_lut(image)
    
    # Apply power-law transformation using LUT with gamma = 2.0
    gamma_corrected_image = power_law_lut(image, gamma=2.0)
    
    # Apply thresholding using LUT with a threshold value of 128
    thresholded_image = threshold_lut(image, threshold_value=128)
    
    # Display original and transformed images
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    
    # Histogram normalized image
    plt.subplot(2, 2, 2)
    plt.title("Histogram Normalized Image")
    plt.imshow(normalized_image, cmap='gray', vmin=0, vmax=255)
    
    # Power-law transformed image
    plt.subplot(2, 2, 3)
    plt.title("Power-law Transformed Image (Gamma = 2.0)")
    plt.imshow(gamma_corrected_image, cmap='gray', vmin=0, vmax=255)
    
    # Thresholded image
    plt.subplot(2, 2, 4)
    plt.title("Thresholded Image (Threshold = 128)")
    plt.imshow(thresholded_image, cmap='gray', vmin=0, vmax=255)
    
    plt.show()

# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print("Usage: python script.py <image_path>")
    sys.exit(1)

# Run the main function with the image path from command-line argument
image_path = sys.argv[1]
main(image_path)
