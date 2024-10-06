import sys
import numpy as np
from libtiff import TIFF
from matplotlib import pyplot as plt

def load_image(image_path):
    tifImg = TIFF.open(image_path)
    image = TIFF.read_image(tifImg)
    tifImg.close()
    return image

def compute_threshold(image, threshold_value=128):
    binary_image = np.where(image > threshold_value, 255, 0)
    return binary_image

def apply_threshold(image, threshold_value=128):
    thresholded_image = compute_threshold(image, threshold_value)
    thresholded_image = thresholded_image.astype(np.uint8)
    return thresholded_image

def main(image_path, threshold_value):
    image = load_image(image_path)
    
    thresholded_image = apply_threshold(image, threshold_value)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    
    plt.subplot(1, 2, 2)
    plt.title(f"Thresholded Image (Threshold = {threshold_value})")
    plt.imshow(thresholded_image, cmap='gray', vmin=0, vmax=255)
    
    plt.show()

if len(sys.argv) != 3:
    print("Usage: python3 script.py <image_path> <threshold_value>")
    sys.exit(1)

image_path = sys.argv[1]
threshold_value = int(sys.argv[2])
main(image_path, threshold_value)
