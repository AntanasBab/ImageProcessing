import sys
import numpy as np
from libtiff import TIFF
from matplotlib import pyplot as plt

def load_tiff_image(tiff_path):
    tif = TIFF.open(tiff_path)
    image = tif.read_image()
    tif.close()
    return image

def linear_transform_image(image, linear_matrix, output_shape):
     # Fill with 0s to start
    transformed_image = np.zeros(output_shape)

    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            original_coords = np.dot(linear_matrix, [i, j])
            x, y = original_coords[0], original_coords[1]

            # Check if the transformed coordinates are within the bounds of image
            if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                transformed_image[i, j] = get_nearest_neighbor(image, x-1, y-1)
    
    return transformed_image

def affine_transform_image(image, affine_matrix, output_shape):
    transformed_image = np.zeros(output_shape)
    inverse_matrix = np.linalg.inv(affine_matrix)

    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            original_coords = np.dot(inverse_matrix, [i, j, 1])
            x, y = original_coords[0], original_coords[1]

            if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                transformed_image[i, j] = get_nearest_neighbor(image, x-1, y-1)
    
    return transformed_image

def get_nearest_neighbor(image, x, y):
    x_rounded = int(round(x))
    y_rounded = int(round(y))
    return image[x_rounded, y_rounded]

if len(sys.argv) != 2:
    print("Usage: python3 CompositeImageTransform.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
image = load_tiff_image(image_path)

linear_matrix = np.array([[1.5, 0], [0, 1.5]])
transformed_image1 = linear_transform_image(image, linear_matrix, (image.shape[0], image.shape[1]))
affine_matrix = np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
transformed_image2 = affine_transform_image(transformed_image1, affine_matrix, (image.shape[0], image.shape[1]))


fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(image, cmap='gray', vmin=0, vmax=255)
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(transformed_image1, cmap='gray', vmin=0, vmax=255)
axs[1].set_title('First transformation')
axs[1].axis('off')

axs[2].imshow(transformed_image2, cmap='gray', vmin=0, vmax=255)
axs[2].set_title('Final image')
axs[2].axis('off')

plt.show()