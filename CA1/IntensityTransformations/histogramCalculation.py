import sys
import numpy as np
from libtiff import TIFF
from matplotlib import pyplot as plt

def load_image(image_path):
    tifImg = TIFF.open(image_path)
    image = TIFF.read_image(tifImg)
    tifImg.close()
    return image

def compute_histogram(image):
    # .ravel to flatten the array
    histogram = np.bincount(image.ravel(), minlength=256)
    return histogram

def plot_histogram(histogram):
    plt.figure(figsize=(10, 5))
    plt.bar(range(256), histogram, color='gray')
    plt.title("Grayscale Intensity Histogram")
    plt.xlabel("Intensity Value")
    plt.ylabel("Pixel Count")
    plt.show()

def main(image_path):
    image = load_image(image_path)
    histogram = compute_histogram(image)
    plot_histogram(histogram)

if len(sys.argv) != 2:
    print("Usage: python histogramCalculation.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
main(image_path)
