from matplotlib import pyplot as plt
import numpy as np
import sys
from common.loadImage import load_tif_image
from common.avgFilter import averaging_kernel, apply_averaging_filter
from common.otsuThreshold import apply_dual_otsu_threshold
from common.CCL import connected_component_labeling

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

def color_water_pixels(labeled_image, bottles_binary, num_labels, ratios, error_threshold=0.25, warning_threshold=0.18):
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
            error_threshold=0.25
            warning_threshold=0.18

        # Blur
        kernel = averaging_kernel(size=13)
        blurred_image = apply_averaging_filter(image, kernel)

        # Dual-Otsu
        bottles_binary, t1, t2 = apply_dual_otsu_threshold(blurred_image)

        # Whole bottles for labelling
        whole_bottles_binary = np.array(bottles_binary)
        whole_bottles_binary[whole_bottles_binary == 2] = 1

        # Segment bottles and label them
        labeled_image, num_labels = connected_component_labeling(whole_bottles_binary)
        print(f"Detected {num_labels} bottles.")

        # Calculate Bottle-to-Water pixel ratios
        ratios = calculate_ratios(labeled_image, bottles_binary, num_labels)

        colored_image = color_water_pixels(labeled_image, bottles_binary, num_labels, ratios, error_threshold, warning_threshold)

        # Output ratios
        for i, ratio in enumerate(ratios, start=1):
            print(f"Bottle {i}: Bottle-to-Water Pixel Ratio = {ratio:.2f}")

        fig, axs = plt.subplots(2, 3, figsize=(10, 10))
        axs[0, 0].imshow(image, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
        axs[0, 0].set_title("Original Bottle Image")
        axs[0, 0].axis('off')

        axs[0, 1].imshow(blurred_image, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
        axs[0, 1].set_title("Blurred Bottle Image")
        axs[0, 1].axis('off')

        axs[0, 2].imshow(bottles_binary, cmap="gray")
        axs[0, 2].set_title("Otsu dual Thresholded Image")
        axs[0, 2].axis('off')

        axs[1, 0].imshow(whole_bottles_binary, cmap="gray")
        axs[1, 0].set_title("Whole Bottles Image")
        axs[1, 0].axis('off')

        axs[1, 1].imshow(labeled_image, cmap="gray")
        axs[1, 1].set_title("Labelled Bottles Image")
        axs[1, 1].axis('off')

        axs[1, 2].imshow(colored_image, cmap="gray")
        axs[1, 2].set_title("Colored Bottles Image")
        axs[1, 2].axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("Usage: python3 bottles.py <bottles_file_path> <error_threshold> <warning_threshold>")
        sys.exit(1)