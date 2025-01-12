import sys
from common.loadImage import load_tif_image
from matplotlib import pyplot as plt
import numpy as np

def median_filter(image, kernel_size=3):
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='edge')

    denoised_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            kernel_window = padded_image[i:i + kernel_size, j:j + kernel_size]
            denoised_image[i, j] = np.median(kernel_window)

    return denoised_image


def analyze_image(circuit_board_file):
    curcuit_board_image = load_tif_image(circuit_board_file)
    
    plt.imshow(curcuit_board_image, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    plt.title("Original Curcuit Board Image")
    plt.show()

    curcuit_board_smoothed = median_filter(curcuit_board_image, 3)
    
    plt.imshow(curcuit_board_smoothed, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    plt.title("Smoothed Curcuit Board Image")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 circuit-board.py <curcuit_board_file_path>")
        sys.exit(1)

    analyze_image(sys.argv[1])
