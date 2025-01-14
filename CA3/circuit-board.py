from common.loadImage import load_tif_image
from common.medianFilter import median_filter
from common.otsuThreshold import (
    otsu_threshold,
    threshold_image,
)
from matplotlib import pyplot as plt
import numpy as np

ROUND_SOLDERING_CENTER_LIST = [[10, 59], [36, 59], [61, 58], [12, 87], [36, 86], [60, 88], 
 [23, 116], [48, 115], [22, 144], [49, 143], [187, 42], [188, 73], 
 [220, 32], [221, 59], [245, 30], [245, 60], [246, 111], [245, 139]]

SQUARE_SOLDERING_CENTER_LIST = [[180, 249], [253, 248]]

BOARD_CONNECTOR_CENTER_LIST = [[8, 5], [20, 4], [32, 5], [45, 5], [58, 5], [70, 5], [83, 5], 
 [95, 5], [108, 5], [120, 5], [133, 5], [145, 5], [158, 5], 
 [170, 5], [183, 5], [196, 5], [208, 5], [220, 5], [233, 5], 
 [245, 5], [258, 5], [270, 5], [282, 5], [295, 5]]

CHIP_CONNECTOR_CENTER_LIST = [[96, 120], [96, 139], [97, 159], [96, 180], [120, 202], [140, 203], 
 [160, 202], [179, 202], [202, 180], [202, 161], [202, 140], [202, 120], 
 [180, 97], [160, 97], [140, 97], [120, 97]]

def is_connected(point, binary_image):
    """Check if a point is connected to any non-zero pixel in the binary image."""
    x, y = point
    if binary_image[y, x] == 1:  # Check if the point itself is part of the object
        return True
    # Check 8-connectivity
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            if (0 <= x + dx < binary_image.shape[1] and
                    0 <= y + dy < binary_image.shape[0] and
                    binary_image[y + dy, x + dx] == 1):
                return True
    return False

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

    plt.figure(figsize=(12, 6))

    plt.imshow(circuit_board_binary, cmap='gray')
    plt.title("Otsu Thresholded Circuit Board Image")
    plt.axis('off')  # Turn off axis for better visual appeal

    # Create a color image to mark missing connections
    marked_image = np.stack((circuit_board_binary,) * 3, axis=-1)  # Convert to RGB

    # Check connectivity for ROUND and SQUARE soldering centers
    all_soldering_centers = (ROUND_SOLDERING_CENTER_LIST +
                             SQUARE_SOLDERING_CENTER_LIST)
    connected_centers = []
    missing_centers = []

    for center in all_soldering_centers:
        if is_connected(center, circuit_board_binary):
            connected_centers.append(center)
        else:
            missing_centers.append(center)
            # Mark the missing center in red
            x, y = center
            if 0 <= x < marked_image.shape[1] and 0 <= y < marked_image.shape[0]:
                marked_image[y, x] = [1, 0, 0]  # Red color in RGB

    plt.imshow(marked_image, interpolation="nearest", vmin=0, vmax=255)
    plt.title("Marked Missing Connections")
    plt.axis('off')  # Turn off axis for better visual appeal
    plt.show()

    print("Connected Soldering Centers:", connected_centers)
    print("Missing Soldering Centers:", missing_centers)

if __name__ == "__main__":
    analyze_image("/home/antbab/ImageProcessing/CA3/ImgSetCa3/imgsetca3/pcb-xray.tif")
