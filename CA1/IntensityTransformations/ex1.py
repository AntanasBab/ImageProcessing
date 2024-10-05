import sys
import numpy as np
from libtiff import TIFF
from matplotlib import pyplot as plt

if len(sys.argv) != 3:
    print("Usage: python3 ex1.py <image_path> <gamma>")
    sys.exit(1)

image_path = sys.argv[1]
gamma = float(sys.argv[2])

tifImg = TIFF.open(image_path)
image = TIFF.read_image(tifImg)
normalized_image = image / 255.0
transformed_image = np.power(normalized_image, gamma)
transformed_image_rescaled = (transformed_image * 255).astype(np.uint8)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray', vmin=0, vmax=255)

plt.subplot(1, 2, 2)
plt.title(f"Power Law Transformed (γ = {gamma})")
plt.imshow(transformed_image_rescaled, cmap='gray', vmin=0, vmax=255)

plt.show()
tifImg.close()
