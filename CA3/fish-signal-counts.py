from libtiff import TIFF
from matplotlib import pyplot as plt
import numpy as np
import sys

# Load the .tif images
def load_tif_image(filename):
    tif = TIFF.open(filename, mode='r')
    image = tif.read_image()
    return np.array(image, dtype=np.float32)

# Create an averaging filter (mean filter)
def averaging_kernel(size):
    """
    Generate a 2D averaging kernel.
    :param size: Size of the kernel (must be odd).
    """
    kernel = np.ones((size, size), dtype=np.float32)
    kernel /= np.sum(kernel)
    return kernel

# Convolution with averaging kernel
def apply_averaging_filter(image, kernel):
    """
    Apply an averaging filter to an image.
    :param image: Input 2D image.
    :param kernel: 2D averaging kernel.
    """
    padded_image = np.pad(image, kernel.shape[0] // 2, mode='reflect')
    smoothed_image = np.zeros_like(image)
    k = kernel.shape[0] // 2
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            smoothed_image[i, j] = np.sum(region * kernel)
    
    return smoothed_image

# Otsu's thresholding
def otsu_threshold(image):
    """
    Compute Otsu's threshold for a given grayscale image.
    """
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
        
        # Calculate between-class variance
        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if variance_between > max_variance:
            max_variance = variance_between
            threshold = t
    
    return threshold

# Thresholding function
def threshold_image(image, threshold):
    return (image > threshold).astype(np.uint8)

# Find connected components using a simple flood-fill algorithm
def find_connected_components(binary_image):
    labels = np.zeros_like(binary_image, dtype=np.int32)
    label = 0
    stack = []
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] == 1 and labels[i, j] == 0:
                label += 1
                stack.append((i, j))
                while stack:
                    x, y = stack.pop()
                    if labels[x, y] == 0:
                        labels[x, y] = label
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < binary_image.shape[0] and 0 <= ny < binary_image.shape[1]:
                                if binary_image[nx, ny] == 1 and labels[nx, ny] == 0:
                                    stack.append((nx, ny))
    return labels, label

# Count signals within a region
def count_signals(region_mask, intensity_image, threshold):
    return np.sum(intensity_image[region_mask] > threshold)

def plot_labeled_image(labels, num_labels):
    """
    Plot the labeled image with cell indices, avoiding black color for the background.
    """
    plt.figure(figsize=(10, 8))
    labeled_image = np.zeros_like(labels, dtype=np.float32)
    
    # Assign unique values to each label
    for label in range(1, num_labels + 1):
        labeled_image[labels == label] = label
    
    # Create a custom colormap that keeps the background black
    from matplotlib.colors import ListedColormap
    cmap = plt.cm.nipy_spectral
    cmap_colors = cmap(np.linspace(0, 1, num_labels + 1))
    cmap_colors[0] = [0, 0, 0, 1.0]  # Background (label 0) is black
    custom_cmap = ListedColormap(cmap_colors)
    
    # Plot the labeled image
    plt.imshow(labeled_image, cmap=custom_cmap)
    plt.colorbar(ticks=np.arange(num_labels + 1), label="Cell Index")
    plt.title("Labeled Cells with Indices")
    plt.show()

# Main analysis function
def analyze_images_with_smoothing(acridine_file, fitc_file, dapi_file):
    # Load images
    acridine = load_tif_image(acridine_file)
    fitc = load_tif_image(fitc_file)
    dapi = load_tif_image(dapi_file)
    
    rgb_image_original = np.stack([acridine, fitc, dapi], axis=-1)
    rgb_image_original = rgb_image_original / np.max(rgb_image_original)  # Normalize for visualization
    plt.imshow(rgb_image_original)
    plt.title("Original RGB Image (Acridine=Red, FITC=Green, DAPI=Blue)")
    plt.show()

    # Create averaging kernel for smoothing
    kernel = averaging_kernel(size=5)  # 5x5 averaging filter
    
    # Apply averaging filter to images
    dapi_smoothed = apply_averaging_filter(dapi, kernel)
    
    # Create RGB image for visualization
    rgb_image = np.stack([acridine, fitc, dapi_smoothed], axis=-1)
    rgb_image = rgb_image / np.max(rgb_image)  # Normalize for visualization
    plt.imshow(rgb_image)
    plt.title("Smoothed RGB Image (Acridine=Red, FITC=Green, DAPI=Blue)")
    plt.show()
    
    # Apply Otsu's thresholding to segment DAPI cells
    dapi_threshold = otsu_threshold(dapi_smoothed)
    dapi_binary = threshold_image(dapi_smoothed, dapi_threshold)
    
    # Plot thresholded DAPI binary image
    plt.figure()
    plt.imshow(dapi_binary, cmap='gray')
    plt.title("Thresholded DAPI Image (Binary, Smoothed)")
    plt.show()
    
    # Find connected components
    labels, num_labels = find_connected_components(dapi_binary)
    
    # Plot labeled cells with indices
    plot_labeled_image(labels, num_labels)
    
    # Analyze each cell
    results = []
    acridine_threshold = otsu_threshold(acridine)
    fitc_threshold = otsu_threshold(fitc)
    
    for label in range(1, num_labels + 1):
        cell_mask = labels == label
        acridine_count = count_signals(cell_mask, acridine, acridine_threshold)
        fitc_count = count_signals(cell_mask, fitc, fitc_threshold)
        ratio = acridine_count / fitc_count if fitc_count > 0 else 0
        results.append({
            "Cell ID": label,
            "Acredine Count": acridine_count,
            "FITC Count": fitc_count,
            "Acredine/FITC Ratio": ratio
        })
    
    # Print results
    print("Cell Analysis Results:")
    for result in results:
        print(f"Cell {result['Cell ID']}: Acredine={result['Acredine Count']}, FITC={result['FITC Count']}, Ratio={result['Acredine/FITC Ratio']:.2f}")
    return results

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 fish-signal-counts.py <acridine_file_path> <dapi_file_path> <fitc_file_path>")
        sys.exit(1)

    # Run the analysis
    analyze_images_with_smoothing(sys.argv[1], sys.argv[3], sys.argv[2])
