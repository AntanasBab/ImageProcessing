import numpy as np;

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