from libtiff import TIFF
import matplotlib.pyplot as plt
import sys

def viewSubImages(file_path):
    tifImg = TIFF.open(file_path)
    directories = list(TIFF.iter_images(tifImg))
    num_images = len(directories)

    fig, axs = plt.subplots(1, num_images, figsize=(15, 15))
    if num_images == 1:
        axs = [axs]
    
    for idx, image in enumerate(directories):
        axs[idx].imshow(image, interpolation='nearest', vmin=0, vmax=255)
        axs[idx].set_title(f'Subimage {idx+1}')
        axs[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

if len(sys.argv) != 2:
    print("Usage: python3 displayAllSubImages.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

viewSubImages(image_path)
