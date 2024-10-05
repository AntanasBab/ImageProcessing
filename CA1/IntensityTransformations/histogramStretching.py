import sys
import numpy as np
from libtiff import TIFF
from matplotlib import pyplot as plt

def load_image(image_path):
    tifImg = TIFF.open(image_path)
    image = TIFF.read_image(tifImg)
    tifImg.close()
    return image

def compute_stretch(image, s_min=0, s_max=255):
    r_min = np.min(image)
    r_max = np.max(image)
    
    if r_max == r_min:
        return np.full(image.shape, s_min if r_max < 128 else s_max, dtype=np.uint8)
    
    stretched = (image - r_min) / (r_max - r_min) * (s_max - s_min) + s_min
    return stretched

def apply_stretch(image):
    stretched_image = compute_stretch(image)
    stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)
    return stretched_image

def main(image_path):
    image = load_image(image_path)
    stretched_image = apply_stretch(image)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    
    plt.subplot(1, 2, 2)
    plt.title("Stretched Image")
    plt.imshow(stretched_image, cmap='gray', vmin=0, vmax=255)
    
    plt.show()

if len(sys.argv) != 2:
    print("Usage: python3 ex2.py <image_path>")
    sys.exit(1)

main(sys.argv[1])
