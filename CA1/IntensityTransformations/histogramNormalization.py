import sys
import numpy as np
from libtiff import TIFF
from matplotlib import pyplot as plt

def load_image(image_path):
    tifImg = TIFF.open(image_path)
    image = TIFF.read_image(tifImg)
    tifImg.close()
    return image

def compute_histogram_normalization(image):
    flat_image = image.ravel()
    
    histogram, bin_edges = np.histogram(flat_image, bins=256, range=(0, 255))
    
    cdf = histogram.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)

    equalized_image = cdf_normalized[flat_image]
    equalized_image = equalized_image.reshape(image.shape)
    
    return equalized_image

def apply_histogram_normalization(image_path):
    original_image = load_image(image_path)
    normalized_image = compute_histogram_normalization(original_image)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    
    plt.subplot(1, 2, 2)
    plt.title("Histogram Normalized Image")
    plt.imshow(normalized_image, cmap='gray', vmin=0, vmax=255)
    
    plt.show()

if len(sys.argv) != 2:
    print("Usage: python3 histogramNormalization.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
apply_histogram_normalization(image_path)
