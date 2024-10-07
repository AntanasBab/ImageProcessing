import sys
import numpy as np
from libtiff import TIFF
from matplotlib import pyplot as plt
from imageLoading import load_image,convert_8bit_to_float,convert_float_to_8bit

def blur_image(image_float):
    kernel = np.array([[1/9, 1/9, 1/9],
                       [1/9, 1/9, 1/9],
                       [1/9, 1/9, 1/9]])

    height, width = image_float.shape
    blurred_image = np.zeros_like(image_float)

    for i in range(1, height-1):
        for j in range(1, width-1):
            region = image_float[i-1:i+2, j-1:j+2]
            blurred_image[i, j] = np.sum(region * kernel)

    return blurred_image

def main(image_path):
    image_float = convert_8bit_to_float(load_image(image_path))
    blurred_image_float = blur_image(image_float)
    blurred_image_8bit = convert_float_to_8bit(blurred_image_float)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_float, cmap='gray', vmin=0, vmax=255)

    plt.subplot(1, 2, 2)
    plt.title("Blurred 8-bit Image")
    plt.imshow(blurred_image_8bit, cmap='gray', vmin=0, vmax=255)

    plt.show()


if len(sys.argv) != 2:
    print("Usage: python blurImage.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
main(image_path)
