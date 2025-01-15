from common.loadImage import load_tif_image
from common.medianFilter import median_filter
from common.otsuThreshold import (
    otsu_threshold,
    threshold_image,
)
from matplotlib import pyplot as plt
import sys

ROUND_SOLDERING_CENTERS = [[10, 59], [36, 59], [61, 58], [12, 87], [36, 86], [60, 88], 
 [23, 116], [48, 115], [22, 144], [49, 143], [187, 42], [188, 73], 
 [220, 32], [221, 59], [245, 30], [245, 60], [246, 111], [245, 139]]

SQUARE_SOLDERING_CENTERS = [[180, 249], [253, 248]]

BOARD_CONNECTOR_CENTERS = [[8, 5], [20, 4], [32, 5], [45, 5], [58, 5], [70, 5], [83, 5], 
 [95, 5], [108, 5], [120, 5], [133, 5], [145, 5], [158, 5], 
 [170, 5], [183, 5], [196, 5], [208, 5], [220, 5], [233, 5], 
 [245, 5], [258, 5], [270, 5], [282, 5], [295, 5]]

CHIP_CONNECTOR_CENTERS = [[96, 120], [96, 139], [97, 159], [96, 180], [120, 202], [140, 203], 
 [160, 202], [179, 202], [202, 180], [202, 161], [202, 140], [202, 120], 
 [180, 97], [160, 97], [140, 97], [120, 97]]

def analyze_image(circuit_board_file):
    circuit_board_image = load_tif_image(circuit_board_file)
    
    plt.imshow(circuit_board_image, cmap="gray", interpolation="nearest",
               vmin=0, vmax=255)
    plt.title("Original Circuit Board Image")
    plt.show()

    circuit_board_filtered = median_filter(circuit_board_image, 3)
    
    plt.imshow(circuit_board_filtered, cmap="gray", interpolation="nearest",
               vmin=0, vmax=255)
    plt.title("Filtered Circuit Board Image")
    plt.show()

    circuit_board_threshold = otsu_threshold(circuit_board_filtered)
    circuit_board_binary = threshold_image(circuit_board_filtered,
                                           circuit_board_threshold)

    plt.imshow(circuit_board_binary, cmap='gray')
    plt.title("Otsu Thresholded Circuit Board Image")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 circuit-board.py <circuit_board_file_path>")
        sys.exit(1)

    analyze_image(sys.argv[1])
