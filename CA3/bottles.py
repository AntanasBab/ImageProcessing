from matplotlib import pyplot as plt
import numpy as np
import sys
from common.loadImage import load_tif_image
from common.avgFilter import averaging_kernel, apply_averaging_filter
from common.otsuThreshold import apply_dual_otsu_threshold
from common.CCL import connected_component_labeling

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

SOBEL_Y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

def apply_sobel(image):
    height, width = image.shape
    edge_magnitude = np.zeros_like(image, dtype=float)
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = image[i-1:i+2, j-1:j+2]
            
            grad_x = np.sum(region * SOBEL_X)
            grad_y = np.sum(region * SOBEL_Y)
            edge_magnitude[i, j] = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize
    edge_magnitude = np.clip(edge_magnitude * 255.0 / np.max(edge_magnitude), 0, 255).astype(np.uint8)
    
    return edge_magnitude

def find_bottleneck_from_bottom(sobel_edges, thresholdQuotient=0.9):
    height, width = sobel_edges.shape

    bottleneck_base_rows = []

    for col in range(width):
        for row in range(height - 1, 1, -1):
            if row - 1 >= 0 and col - 1 >= 0 and col + 1 < width:
                area = sobel_edges[row - 1:row + 2, col - 1:col + 2]
                intensity_changes = np.abs(np.diff(area, axis=0))

                if np.any(intensity_changes > 255 * thresholdQuotient):
                    bottleneck_base_rows.append(row)
                    break
        else:
            bottleneck_base_rows.append(-1)

    valid_rows = sorted([row for row in bottleneck_base_rows if row >= 0])
    bottleneck_base_row_max = np.median(valid_rows) if valid_rows else -1

    return bottleneck_base_row_max

def calculate_ratios(labeled_image, bottles_binary, num_labels):
    ratios = []
    for label in range(1, num_labels + 1):
        bottle_mask = labeled_image == label
        bottle_pixels = bottles_binary[bottle_mask]

        water_pixels = np.sum(bottle_pixels == 1)
        bottle_pixels = np.sum(bottle_pixels == 2)

        if bottle_pixels > 0:
            ratio = bottle_pixels / water_pixels
        else:
            ratio = 0 

        ratios.append(ratio)

    return ratios

def color_water_pixels(labeled_image, bottles_binary, num_labels, ratios, error_threshold, warning_threshold):
    colored_image = np.zeros((bottles_binary.shape[0], bottles_binary.shape[1], 3), dtype=np.uint8)
    
    for label in range(1, num_labels + 1):
        bottle_mask = labeled_image == label
        
        ratio = ratios[label - 1]

        if ratio >= error_threshold:
            color = [200, 0, 0]  # Red
        elif ratio >= warning_threshold:
            color = [200, 200, 0]  # Yellow
        else:
            color = [0, 200, 0]  # Green
        
        colored_image[bottle_mask & (bottles_binary == 1)] = color

    return colored_image

if __name__ == "__main__":
    if len(sys.argv) == 2 or len(sys.argv) == 4:
        image = load_tif_image(sys.argv[1])

        if len(sys.argv) == 4:
            error_threshold = float(sys.argv[2])
            warning_threshold = float(sys.argv[3])

            if error_threshold <= warning_threshold:
                print('Error threshold should be smaller than warning threshold')
                sys.exit(1)
        else:
            error_threshold=0.15
            warning_threshold=0.05

        # Blur
        kernel = averaging_kernel(size=9)
        blurred_image = apply_averaging_filter(image, kernel)

        # Dual-Otsu
        bottles_binary, t1, t2 = apply_dual_otsu_threshold(blurred_image)

        # Whole bottles for labelling
        whole_bottles_binary = np.array(bottles_binary)
        whole_bottles_binary[whole_bottles_binary == 2] = 1

        # Get shape edges
        sobel_edges = apply_sobel(whole_bottles_binary)

        # Detect the top of the bottleneck
        bottleneck_top_row = int(find_bottleneck_from_bottom(sobel_edges))

        # Segment bottles and label them
        labeled_image, num_labels = connected_component_labeling(whole_bottles_binary[bottleneck_top_row - 1:,:])
        print(f"Detected {num_labels} bottles.")

        # Calculate Bottle-to-Water pixel ratios
        ratios = calculate_ratios(labeled_image, bottles_binary[bottleneck_top_row - 1:,:], num_labels)
        colored_image = color_water_pixels(labeled_image, bottles_binary[bottleneck_top_row - 1:,:], num_labels, ratios, error_threshold, warning_threshold)

        fig, axs = plt.subplots(2, 4, figsize=(15, 10))
        axs[0, 0].imshow(image, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
        axs[0, 0].set_title("Original Bottle Image")
        axs[0, 0].axis('off')

        axs[0, 1].imshow(blurred_image, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
        axs[0, 1].set_title("Blurred Bottle Image")
        axs[0, 1].axis('off')

        axs[0, 2].imshow(bottles_binary, cmap="gray")
        axs[0, 2].set_title("Otsu dual Thresholded Image")
        axs[0, 2].axis('off')

        axs[0, 3].imshow(whole_bottles_binary, cmap="gray")
        axs[0, 3].set_title("Whole Bottles Image")
        axs[0, 3].axis('off')

        axs[1, 0].imshow(sobel_edges, cmap="gray")
        axs[1, 0].set_title("Sobel Edges Image")
        axs[1, 0].axis('off')

        axs[1, 1].imshow(sobel_edges, cmap="gray")
        axs[1, 1].axhline(y=bottleneck_top_row, color='r', linestyle='-', linewidth=1, label="Bottleneck Top (Red)")
        axs[1, 1].set_title("Sobel Edges with Bottleneck Line")
        axs[1, 1].axis('off')

        axs[1, 2].imshow(labeled_image, cmap="gray")
        axs[1, 2].set_title("Labelled Bottles Image")
        axs[1, 2].axis('off')

        axs[1, 3].imshow(colored_image, cmap="gray")
        axs[1, 3].set_title("Colored Bottles Image")
        axs[1, 3].axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("Usage: python3 bottles.py <bottles_file_path> <error_threshold> <warning_threshold>")
        sys.exit(1)