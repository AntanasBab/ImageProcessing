import numpy as np
import matplotlib.pyplot as plt
from libtiff import TIFF
import sys

def load_image(image_path):
    tifImg = TIFF.open(image_path)
    image = tifImg.read_image()
    tifImg.close()
    return image

def pad_image(image, M, N):
    padded_image = np.zeros((2 * M, 2 * N), dtype=np.complex128) 

    for i in range(M): 
        for j in range(N): 
            padded_image[i, j] = image[i, j] 
    return padded_image

def shift_image(image, M, N):
    x = np.arange(2 * M)
    y = np.arange(2 * N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    shifted_image = image * ((-1) ** (X + Y))

    return (X, Y, shifted_image)

def unshift_image(image, X, Y):
    return image * ((-1) ** (X + Y))

def fourier_transform(image):
    return np.fft.fft2(image)

def inverse_fourier_transform(image):
    return np.fft.ifft2(image)

def crop_image(image, M, N):
    return image[:M, :N]

def process_image(filename):
    image = load_image(filename)
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 3, 1)
    plt.title("1. Original Image (MxN)")
    plt.imshow(image, cmap='gray', interpolation='nearest', vmax=255, vmin=0)

    M, N = image.shape
    padded_image = pad_image(image, M, N)  

    plt.subplot(2, 3, 2)
    plt.title("2. Padded Image (2Mx2N)")
    plt.imshow(np.abs(padded_image), cmap='gray', interpolation='nearest', vmax=255, vmin=0)

    X, Y, shifted_image = shift_image(padded_image, M, N)

    dimming_factor = 0.5
    shifted_image_dimmer = shifted_image * dimming_factor
    shifted_image_display = np.abs(shifted_image_dimmer)

    plt.subplot(2, 3, 3)
    plt.title("3. Shifted Image for Periodicity")
    plt.imshow(shifted_image_display, cmap='gray', interpolation='nearest', vmax=255, vmin=0)

    dft_image = fourier_transform(shifted_image)
    magnitude_spectrum = np.log(np.abs(dft_image) + 1)

    plt.subplot(2, 3, 4)
    plt.title("4. DFT Magnitude Spectrum")
    plt.imshow(magnitude_spectrum, cmap='gray')

    idft_image = inverse_fourier_transform(dft_image)
    shifted_idft_image = unshift_image(idft_image, X, Y)

    plt.subplot(2, 3, 5)
    plt.title("5. Inverse DFT (Shifted Back)")
    plt.imshow(np.abs(shifted_idft_image), cmap='gray', interpolation='nearest', vmax=255, vmin=0)

    final_image = np.abs(crop_image(shifted_idft_image, M, N))

    plt.subplot(2, 3, 6)
    plt.title("6. Cropped Final Image")
    plt.imshow(final_image, cmap='gray', interpolation='nearest', vmax=255, vmin=0)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 FourierTransform.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    process_image(image_path)
