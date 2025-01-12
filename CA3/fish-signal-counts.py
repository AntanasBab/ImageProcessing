from libtiff import TIFF
from matplotlib import pyplot as plt
import numpy as np
import sys

def load_tif_image(filename):
    tif = TIFF.open(filename, mode='r')
    image = tif.read_image()
    return np.array(image, dtype=np.float32)

def averaging_kernel(size):
    kernel = np.ones((size, size), dtype=np.float32)
    kernel /= np.sum(kernel)
    return kernel

def apply_averaging_filter(image, kernel):
    padded_image = np.pad(image, kernel.shape[0] // 2, mode='reflect')
    smoothed_image = np.zeros_like(image)
    k = kernel.shape[0] // 2
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            smoothed_image[i, j] = np.sum(region * kernel)
    
    return smoothed_image

def otsu_threshold(image):
    pixel_counts = np.bincount(image.astype(int).flatten(), minlength=256)
    total_pixels = np.sum(pixel_counts)
    total_sum = np.sum(np.arange(256) * pixel_counts)
    
    max_variance = 0
    threshold = 0
    weight_background = 0
    sum_background = 0
    
    for t in range(256):
        weight_background += pixel_counts[t]
        weight_foreground = total_pixels - weight_background
        
        if weight_background == 0 or weight_foreground == 0:
            continue
        
        sum_background += t * pixel_counts[t]
        mean_background = sum_background / weight_background
        mean_foreground = (total_sum - sum_background) / weight_foreground
        
        # Calculate between class variance
        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if variance_between > max_variance:
            max_variance = variance_between
            threshold = t
    
    return threshold

def threshold_image(image, threshold):
    return (image > threshold).astype(np.uint8)

def connected_component_labeling(binary_image):
    rows, cols = binary_image.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    label = 1
    equivalences = {}

    # First pass: assign provisional labels and track equivalences
    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] == 1:
                neighbors = []

                for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and labels[ni, nj] > 0:
                        neighbors.append(labels[ni, nj])

                if not neighbors:
                    labels[i, j] = label
                    equivalences[label] = label
                    label += 1
                else:
                    min_label = min(neighbors)
                    labels[i, j] = min_label

                    # Update equivalence table
                    for neighbor_label in neighbors:
                        if neighbor_label != min_label:
                            equivalences[max(min_label, neighbor_label)] = min(min_label, neighbor_label)

    # Resolve equivalences
    for key in sorted(equivalences.keys(), reverse=True):
        root = equivalences[key]
        while root != equivalences[root]:
            root = equivalences[root]
        equivalences[key] = root

    # Second pass: relabel pixels with resolved labels
    resolved_labels = np.zeros_like(labels)
    new_label = 1
    label_map = {}

    for i in range(rows):
        for j in range(cols):
            if labels[i, j] > 0:
                resolved_label = equivalences[labels[i, j]]
                if resolved_label not in label_map:
                    label_map[resolved_label] = new_label
                    new_label += 1
                resolved_labels[i, j] = label_map[resolved_label]

    return resolved_labels, new_label - 1

def find_root(equivalence, label):
    while equivalence[label] != label:
        label = equivalence[label]
    return label

def count_signals(region_mask, intensity_image, threshold=0):
    return np.sum(intensity_image[region_mask] > threshold)

def plot_labeled_image(labels, num_labels):
    plt.figure(figsize=(10, 8))
    labeled_image = np.zeros_like(labels, dtype=np.float32)
    
    for label in range(1, num_labels + 1):
        labeled_image[labels == label] = label
    
    # Create a custom colormap for indexes
    from matplotlib.colors import ListedColormap
    cmap = plt.cm.nipy_spectral
    cmap_colors = cmap(np.linspace(0, 1, num_labels + 1))
    cmap_colors[0] = [0, 0, 0, 1.0]
    custom_cmap = ListedColormap(cmap_colors)
    
    plt.imshow(labeled_image, cmap=custom_cmap)
    plt.colorbar(ticks=np.arange(num_labels + 1), label="Cell Index")
    plt.title("Labeled Cells with Indices")
    plt.show()

def analyze_images(acridine_file, fitc_file, dapi_file):
    acridine = load_tif_image(acridine_file)
    fitc = load_tif_image(fitc_file)
    dapi = load_tif_image(dapi_file)
    
    rgb_image_original = np.stack([acridine, fitc, dapi], axis=-1)
    rgb_image_original = rgb_image_original / np.max(rgb_image_original)  # Normalize for visualization
    plt.imshow(rgb_image_original)
    plt.title("Original RGB Image (Acridine=Red, FITC=Green, DAPI=Blue)")
    plt.show()

    kernel = averaging_kernel(size=5)
    
    dapi_smoothed = apply_averaging_filter(dapi, kernel)
    
    rgb_image = np.stack([acridine, fitc, dapi_smoothed], axis=-1)
    rgb_image = rgb_image / np.max(rgb_image)  # Normalize for visualization
    plt.imshow(rgb_image)
    plt.title("Smoothed RGB Image (Acridine=Red, FITC=Green, DAPI=Blue)")
    plt.show()
    
    dapi_threshold = otsu_threshold(dapi_smoothed)
    dapi_binary = threshold_image(dapi_smoothed, dapi_threshold)
    
    plt.figure()
    plt.imshow(dapi_binary, cmap='gray')
    plt.title("Thresholded DAPI Image (Binary, Smoothed)")
    plt.show()
    
    labels, num_labels = connected_component_labeling(dapi_binary)
    plot_labeled_image(labels, num_labels)
    
    results = []
    for label in range(1, num_labels + 1):
        cell_mask = labels == label
        acridine_count = count_signals(cell_mask, acridine)
        fitc_count = count_signals(cell_mask, fitc)
        ratio = acridine_count / fitc_count if fitc_count > 0 else 0
        results.append({
            "Cell ID": label,
            "Acredine Count": acridine_count,
            "FITC Count": fitc_count,
            "Acredine/FITC Ratio": ratio
        })
    
    # Results
    print("Cell Analysis Results:")
    for result in results:
        print(f"Cell {result['Cell ID']}: Acredine={result['Acredine Count']}, FITC={result['FITC Count']}, Ratio={result['Acredine/FITC Ratio']:.2f}")
    return results

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 fish-signal-counts.py <acridine_file_path> <dapi_file_path> <fitc_file_path>")
        sys.exit(1)

    analyze_images(sys.argv[1], sys.argv[3], sys.argv[2])
