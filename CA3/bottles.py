from libtiff import TIFF
from matplotlib import pyplot as plt
import numpy as np
import sys

def load_image(file_path):
    """Load a TIFF image and return it as a NumPy array."""
    tiff = TIFF.open(file_path, mode='r')
    image = tiff.read_image()
    tiff.close()
    return image

def dual_otsu_threshold(image):
    """Apply dual Otsu's thresholding to segment the image into three classes."""
    histogram, bin_edges = np.histogram(image.flatten(), bins=256, range=(0, 255))
    total_pixels = image.size
    sum_total = np.dot(np.arange(256), histogram)

    weight_background = 0
    weight_foreground = 0
    sum_background = 0
    max_variance = 0
    thresholds = (0, 0)

    for t1 in range(256):
        weight_background += histogram[t1]
        sum_background += t1 * histogram[t1]
        weight_foreground = total_pixels - weight_background
        sum_foreground = sum_total - sum_background

        if weight_background == 0 or weight_foreground == 0:
            continue

        mean_background = sum_background / weight_background
        mean_foreground = sum_foreground / weight_foreground
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if between_class_variance > max_variance:
            max_variance = between_class_variance
            thresholds = (t1, 0)

    # Find the second threshold
    t1 = thresholds[0]
    max_variance = 0

    for t2 in range(t1 + 1, 256):
        weight_background2 = np.sum(histogram[:t2])
        weight_foreground2 = np.sum(histogram[t2:])
        sum_background2 = np.dot(np.arange(t2), histogram[:t2])
        sum_foreground2 = sum_total - sum_background2

        if weight_background2 == 0 or weight_foreground2 == 0:
            continue

        mean_background2 = sum_background2 / weight_background2
        mean_foreground2 = sum_foreground2 / weight_foreground2
        between_class_variance2 = weight_background2 * weight_foreground2 * (mean_background2 - mean_foreground2) ** 2

        if between_class_variance2 > max_variance:
            max_variance = between_class_variance2
            thresholds = (t1, t2)

    return thresholds

def segment_image(image, thresholds):
    """Segment the image using the dual Otsu thresholds."""
    t1, t2 = thresholds
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    segmented_image[image <= t1] = 0  # Background
    segmented_image[(image > t1) & (image <= t2)] = 1  # Liquid
    segmented_image[image > t2] = 2  # Bottle body
    return segmented_image

def connected_component_labeling(binary_image):
    """Perform Connected Component Labeling (CCL) to detect bottle regions."""
    labeled_image = np.zeros_like(binary_image, dtype=np.int32)
    label = 1
    equivalences = {}

    # First pass: Assign preliminary labels and track equivalences
    for i in range(1, binary_image.shape[0] - 1):
        for j in range(1, binary_image.shape[1] - 1):
            if binary_image[i, j] == 1:
                neighbors = [labeled_image[i-1, j], labeled_image[i, j-1]]
                neighbors = [label for label in neighbors if label > 0]

                if not neighbors:
                    labeled_image[i, j] = label
                    equivalences[label] = label
                    label += 1
                else:
                    min_label = min(neighbors)
                    labeled_image[i, j] = min_label
                    for neighbor in neighbors:
                        equivalences[neighbor] = min_label

    # Second pass: Resolve equivalences
    for i in range(labeled_image.shape[0]):
        for j in range(labeled_image.shape[1]):
            if labeled_image[i, j] > 0:
                labeled_image[i, j] = equivalences[labeled_image[i, j]]

    return labeled_image, label - 1

def analyze_bottles(image, labeled_image, num_labels):
    """Analyze each labeled region to determine liquid level and classify."""
    results = []
    for label in range(1, num_labels + 1):
        # Extract the region corresponding to this label
        mask = (labeled_image == label)
        if not mask.any():
            # Skip this label if the mask is empty
            continue

        rows, cols = np.where(mask)
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        bottle_region = image[y_min:y_max+1, x_min:x_max+1]

        # Compute liquid level
        vertical_profile = np.mean(bottle_region, axis=1)  # Mean intensity per row
        liquid_level = y_min + np.argmin(vertical_profile)  # Darkest row (top of liquid)

        # Determine threshold positions
        bottle_height = y_max - y_min + 1
        shoulder_line = y_min + int(bottle_height * 0.6)  # Approx shoulder (60% height)
        midway_line = y_min + int((shoulder_line + y_max) / 2)  # Midway point

        # Classify the bottle
        if liquid_level < midway_line:
            results.append((x_min, x_max, y_min, y_max, "Underfilled", liquid_level))
        else:
            results.append((x_min, x_max, y_min, y_max, "Properly Filled", liquid_level))
    
    return results

def visualize_results(image, segmented_image, labeled_image, results):
    """Visualize all steps and final results."""
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')

    # Segmented image
    plt.subplot(1, 4, 2)
    plt.title("Segmented Image (Dual Otsu)")
    plt.imshow(segmented_image, cmap='gray')

    # Labeled image
    plt.subplot(1, 4, 3)
    plt.title("Labeled Image (CCL)")
    plt.imshow(labeled_image, cmap='nipy_spectral')

    # Final results
    plt.subplot(1, 4, 4)
    plt.title("Final Classification")
    plt.imshow(image, cmap='gray')
    for (x_min, x_max, y_min, y_max, label, liquid_level) in results:
        color = 'red' if label == "Underfilled" else 'green'
        plt.plot([x_min, x_max], [liquid_level, liquid_level], color=color, linewidth=2)
        plt.text(x_min, liquid_level - 10, label, color=color, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()


# Main script
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 circuit-board.py <curcuit_board_file_path>")
        sys.exit(1)

    image = load_image(sys.argv[1])

     # Apply dual Otsu's thresholding
    thresholds = dual_otsu_threshold(image)
    segmented_image = segment_image(image, thresholds)

    # Perform CCL to find bottle regions
    binary_image = (segmented_image == 1).astype(np.uint8)  # Focus on the liquid
    labeled_image, num_labels = connected_component_labeling(binary_image)

    # Analyze and classify bottles
    results = analyze_bottles(image, labeled_image, num_labels)

    # Visualize results
    visualize_results(image, segmented_image, labeled_image, results)
