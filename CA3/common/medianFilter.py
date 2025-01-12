import numpy as np

def median_filter(image, kernel_size=3):
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='edge')

    denoised_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            kernel_window = padded_image[i:i + kernel_size, j:j + kernel_size]
            denoised_image[i, j] = np.median(kernel_window)

    return denoised_image