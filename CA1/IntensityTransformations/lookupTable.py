import sys
import numpy as np
from libtiff import TIFF
from matplotlib import pyplot as plt

def load_image(image_path):
    tifImg = TIFF.open(image_path)
    image = TIFF.read_image(tifImg)
    tifImg.close()
    return image

def create_lookup_table(transformation_func):
    lut = np.array([transformation_func(i) for i in range(256)], dtype=np.uint8)
    return lut

def apply_lut_transformation(image, lut):
    transformed_image = lut[image]
    return transformed_image

def histogram_normalization_lut(image):
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 255))
    cdf = histogram.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())  # Normalize CDF to [0, 255]
    cdf_normalized = cdf_normalized.astype(np.uint8)
    
    lut = create_lookup_table(lambda i: cdf_normalized[i])
    return apply_lut_transformation(image, lut)

def power_law_lut(image, gamma):
    lut = create_lookup_table(lambda i: np.clip(255 * (i / 255) ** gamma, 0, 255))
    return apply_lut_transformation(image, lut)

def threshold_lut(image, threshold_value):
    lut = create_lookup_table(lambda i: 255 if i > threshold_value else 0)
    
    return apply_lut_transformation(image, lut)

def main(image_path):
    image = load_image(image_path)
    
    normalized_image = histogram_normalization_lut(image)
    
    gamma_corrected_image = power_law_lut(image, gamma=2.0)
    
    thresholded_image = threshold_lut(image, threshold_value=128)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)

    plt.subplot(2, 2, 2)
    plt.title("Histogram Normalized Image")
    plt.imshow(normalized_image, cmap='gray', vmin=0, vmax=255)
    
    plt.subplot(2, 2, 3)
    plt.title("Power-law Transformed Image (Gamma = 2.0)")
    plt.imshow(gamma_corrected_image, cmap='gray', vmin=0, vmax=255)
    
    plt.subplot(2, 2, 4)
    plt.title("Thresholded Image (Threshold = 128)")
    plt.imshow(thresholded_image, cmap='gray', vmin=0, vmax=255)
    
    plt.show()

if len(sys.argv) != 2:
    print("Usage: python3 lookupTable.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
main(image_path)
