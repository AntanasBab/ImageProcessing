import numpy as np
import matplotlib.pyplot as plt
import pyfftw  # FFTW3 library for Fourier Transforms
from libtiff import TIFF
import sys

# Step 1: Input image M x N
def read_image(filepath):
    tif = TIFF.open(filepath)
    image = tif.read_image()
    return np.array(image, dtype=np.float32)

# Step 2: Pad image to 2M x 2N
def pad_image(image):
    M, N = image.shape
    padded_image = np.zeros((2 * M, 2 * N), dtype=np.float32)
    padded_image[:M, :N] = image  # Place original image in the top-left corner
    return padded_image

# Step 3: Shift the image for periodicity by multiplying by (-1)^(x + y)
def shift_image_for_periodicity(padded_image):
    M, N = padded_image.shape
    shifted_image = padded_image * np.fromfunction(lambda x, y: (-1) ** (x + y), (M, N))
    return shifted_image

# Step 4: Compute the DFT using FFTW3
def compute_dft(shifted_image):
    fft_input = pyfftw.empty_aligned(shifted_image.shape, dtype='complex64')
    fft_output = pyfftw.empty_aligned(shifted_image.shape, dtype='complex64')
    fft_input[:] = shifted_image
    fft_object = pyfftw.FFTW(fft_input, fft_output, direction='FFTW_FORWARD')
    dft_result = fft_object()
    return dft_result

# Step 5: Visualize the steps
def visualize_steps(original, padded, shifted, dft):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Original image
    axs[0, 0].imshow(original, cmap='gray')
    axs[0, 0].set_title('Original Image')

    # Padded image
    axs[0, 1].imshow(padded, cmap='gray')
    axs[0, 1].set_title('Padded Image (2M x 2N)')

    # Shifted image
    axs[1, 0].imshow(shifted, cmap='gray')
    axs[1, 0].set_title('Shifted Image for Periodicity')

    # Magnitude of DFT result
    axs[1, 1].imshow(np.log(1 + np.abs(dft)), cmap='gray')  # Use log scale for better visibility
    axs[1, 1].set_title('DFT Magnitude')

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Main function to execute all steps
def main(image_path):
    # Step 1: Load the image
    original_image = read_image(image_path)

    # Step 2: Pad the image
    padded_image = pad_image(original_image)

    # Step 3: Shift the image for periodicity
    shifted_image = shift_image_for_periodicity(padded_image)

    # Step 4: Perform the DFT
    dft_result = compute_dft(shifted_image)

    # Step 5: Visualize each step
    visualize_steps(original_image, padded_image, shifted_image, dft_result)

# Example usage
if len(sys.argv) != 2:
    print("Usage: python FourierTransform.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
main(image_path)