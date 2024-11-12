import numpy as np
import matplotlib.pyplot as plt
import sys
from FourierTransformation.FourierTransform import *

def ideal_filter(shape, cutoff, high_pass=False):
    M, N = shape
    X, Y = np.meshgrid(np.arange(-N//2, N//2), np.arange(-M//2, M//2))
    distance = np.sqrt(X**2 + Y**2)
    if high_pass:
        return (distance > cutoff).astype(float)
    else:
        return (distance <= cutoff).astype(float)

def butterworth_filter(shape, cutoff, order=2, high_pass=False):
    M, N = shape
    X, Y = np.meshgrid(np.arange(-N//2, N//2), np.arange(-M//2, M//2))
    distance = np.sqrt(X**2 + Y**2)
    if high_pass:
        return 1 / (1 + (cutoff / (distance + 1e-6)) ** (2 * order))
    else:
        return 1 / (1 + (distance / (cutoff + 1e-6)) ** (2 * order))

def gaussian_filter(shape, cutoff, high_pass=False):
    M, N = shape
    X, Y = np.meshgrid(np.arange(-N//2, N//2), np.arange(-M//2, M//2))
    distance = np.sqrt(X**2 + Y**2)
    if high_pass:
        return 1 - np.exp(-(distance ** 2) / (2 * (cutoff ** 2)))
    else:
        return np.exp(-(distance ** 2) / (2 * (cutoff ** 2)))

def apply_filter(dft_image, filter_mask):
    return dft_image * filter_mask

def process_image(filename):
    image = load_image(filename)
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 3, 1)
    plt.title("1. Original Image (MxN)")
    plt.imshow(image, cmap='gray', interpolation='nearest', vmax=255, vmin=0)

    M, N = image.shape
    padded_image = pad_image(image, M, N)

    X, Y, shifted_image = shift_image(padded_image, M, N)
    dft_image = fourier_transform(shifted_image)
    magnitude_spectrum = np.log(np.abs(dft_image) + 1)

    plt.subplot(3, 3, 2)
    plt.title("2. DFT Magnitude Spectrum")
    plt.imshow(magnitude_spectrum, cmap='gray')

    cutoff = 50
    ideal_low_pass = ideal_filter(dft_image.shape, cutoff, high_pass=False)
    ideal_high_pass = ideal_filter(dft_image.shape, cutoff, high_pass=True)
    butterworth_low_pass = butterworth_filter(dft_image.shape, cutoff, order=2, high_pass=False)
    butterworth_high_pass = butterworth_filter(dft_image.shape, cutoff, order=2, high_pass=True)
    gaussian_low_pass = gaussian_filter(dft_image.shape, cutoff, high_pass=False)
    gaussian_high_pass = gaussian_filter(dft_image.shape, cutoff, high_pass=True)

    filters = [
        ("Ideal Low-Pass", ideal_low_pass),
        ("Ideal High-Pass", ideal_high_pass),
        ("Butterworth Low-Pass", butterworth_low_pass),
        ("Butterworth High-Pass", butterworth_high_pass),
        ("Gaussian Low-Pass", gaussian_low_pass),
        ("Gaussian High-Pass", gaussian_high_pass),
    ]

    for i, (title, filter_mask) in enumerate(filters, start=3):
        filtered_dft = apply_filter(dft_image, filter_mask)
        filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_dft))
        final_image = np.abs(filtered_image[:M, :N])

        plt.subplot(3, 3, i)
        plt.title(title)
        plt.imshow(final_image, cmap='gray', interpolation='nearest', vmax=255, vmin=0)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python FourierTransform.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    process_image(image_path)

