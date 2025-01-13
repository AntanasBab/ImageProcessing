from common.loadImage import load_tif_image
from common.medianFilter import median_filter
from common.otsuThreshold import apply_dual_otsu_threshold, otsu_threshold, threshold_image
from matplotlib import pyplot as plt

ROUND_SOLDERING_CENTER_LIST = [
    [11, 58],
    [36, 58],
    [60, 58],
    [11, 87],
    [36, 87],
    [60, 87],
    [23, 115],
    [49, 115],
    [23, 144],
    [49, 144],
    [187, 44],
    [187, 73],
    [220, 30],
    [220, 59],
    [245, 30],
    [245, 59],
    [245, 111],
    [245, 140],
]

SQUARE_SOLDERING_CENTER_LIST = [[180, 249], [253, 249]]

BOARD_CONNECTOR_CENTER_LIST = [
    [8, 5],
    [20, 5],
    [32, 5],
    [45, 5],
    [58, 5],
    [70, 5],
    [83, 5],
    [95, 5],
    [108, 5],
    [120, 5],
    [133, 5],
    [145, 5],
    [158, 5],
    [170, 5],
    [183, 5],
    [196, 5],
    [208, 5],
    [220, 5],
    [233, 5],
    [245, 5],
    [258, 5],
    [270, 5],
    [282, 5],
    [295, 5],
]

CHIP_CONNECTOR_CENTER_LIST = [
    [96, 120],
    [96, 140],
    [96, 159],
    [96, 180],
    [120, 202],
    [140, 202],
    [160, 202],
    [179, 202],
    [202, 180],
    [202, 160],
    [202, 140],
    [202, 120],
    [180, 97],
    [160, 97],
    [140, 97],
    [120, 97],
]

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
    # if len(sys.argv) != 2:
    #     print("Usage: python3 circuit-board.py <curcuit_board_file_path>")
    #     sys.exit(1)

    analyze_image("/home/antbab/ImageProcessing/CA3/ImgSetCa3/imgsetca3/pcb-xray.tif")
