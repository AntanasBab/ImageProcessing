import sys
from common.loadImage import load_tif_image
from common.medianFilter import median_filter
from common.otsuThreshold import apply_dual_otsu_threshold, otsu_threshold, threshold_image
from matplotlib import pyplot as plt

def analyze_image(circuit_board_file):
    curcuit_board_image = load_tif_image(circuit_board_file)
    
    plt.imshow(curcuit_board_image, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    plt.title("Original Curcuit Board Image")
    plt.show()

    curcuit_board_filtered = median_filter(curcuit_board_image, 3)
    
    plt.imshow(curcuit_board_filtered, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    plt.title("Filtered Curcuit Board Image")
    plt.show()

    curcuit_board_threshold = otsu_threshold(curcuit_board_filtered)
    curcuit_board_binary = threshold_image(curcuit_board_filtered, curcuit_board_threshold)

    curcuit_board_binary2, _, _ = apply_dual_otsu_threshold(curcuit_board_filtered)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(curcuit_board_binary, cmap='gray')
    plt.title("Otsu Thresholded Circuit Board Image")
    plt.axis('off')  # Turn off axis for better visual appeal

    plt.subplot(1, 2, 2)
    plt.imshow(curcuit_board_binary2, cmap='gray')
    plt.title("Dual Thresholded Circuit Board Image")
    plt.axis('off')
    plt.show()
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 circuit-board.py <curcuit_board_file_path>")
        sys.exit(1)

    analyze_image(sys.argv[1])
