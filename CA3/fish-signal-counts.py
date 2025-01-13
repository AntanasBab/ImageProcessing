from matplotlib import pyplot as plt
import numpy as np
import sys
from common.avgFilter import apply_averaging_filter, averaging_kernel
from common.loadImage import load_tif_image
from common.otsuThreshold import otsu_threshold, threshold_image
from common.CCL import connected_component_labeling, plot_labeled_image

def count_signals(region_mask, intensity_image, threshold=0):
    return np.sum(intensity_image[region_mask] > threshold)

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
