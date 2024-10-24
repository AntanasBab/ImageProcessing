import numpy as np
import matplotlib.pyplot as plt
from libtiff import TIFF
import sys

def load_image(image_path):
    tifImg = TIFF.open(image_path)
    image = tifImg.read_image()
    tifImg.close()
    return image

def pad_image(image):
    M, N = image.shape
    padded_image = np.zeros((2 * M, 2 * N), dtype=np.complex128)  # Initialize padded image
    
    for i in range(M):  # Iterate over rows
        for j in range(N):  # Iterate over columns
            padded_image[i, j] = image[i, j]  # Copy the original image pixel by pixel
    
    return padded_image

def shift_image_for_periodicity(image):
    M, N = image.shape
    shifted_image = np.zeros_like(image, dtype=np.complex128)  # Initialize an empty array for the shifted image
    
    for i in range(M):  # Loop over each row
        for j in range(N):  # Loop over each column
            shift_factor = (-1) ** (i + j)  # Compute the shift factor for each element
            shifted_image[i, j] = image[i, j] * shift_factor  # Apply the shift
    
    return shifted_image

def forward_fourier_transform(image):
    return np.fft.fft2(image)

def inverse_fourier_transform(fft_image):
    return np.fft.ifft2(fft_image) 

def apply_periodicity_to_real(ifft_image):
    M, N = ifft_image.shape
    x, y = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')
    shift_factor = (-1) ** (x + y)
    real_part_shifted = np.real(ifft_image) * shift_factor
    return real_part_shifted

def extract_upper_left_quadrant(image):
    M, N = image.shape
    return image[:M // 2, :N // 2]

def process_image(filename):
    original_image = load_image(filename)
    
    padded_image = pad_image(original_image)
    
    shifted_image = shift_image_for_periodicity(padded_image)
    
    # Plot original, padded, and shifted images
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(padded_image), cmap='gray')
    plt.title("Padded Image")
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(shifted_image), cmap='gray')
    plt.title("Shifted Image")
    
    plt.tight_layout()
    plt.show()
    
    fft_image = forward_fourier_transform(shifted_image)
    
    magnitude_spectrum = np.log(1 + np.abs(fft_image))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum of DFT')
    plt.colorbar()
    plt.show()

    ifft_image = inverse_fourier_transform(fft_image)
    real_part_shifted = apply_periodicity_to_real(ifft_image)
    
    upper_left_image = extract_upper_left_quadrant(real_part_shifted)

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(real_part_shifted), cmap='gray')
    plt.title("IDFT with Periodicity Shift")
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(upper_left_image), cmap='gray')
    plt.title("Upper-Left Quadrant of IDFT")
    
    plt.show()

if len(sys.argv) != 2:
    print("Usage: python FourierTransform.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
process_image(image_path)
