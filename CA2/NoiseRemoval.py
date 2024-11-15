import sys
import numpy as np
import matplotlib.pyplot as plt
from FourierTransformation.FourierTransform import load_image

def show_fft(image):
    fft_image = np.fft.fftshift(np.fft.fft2(image))
    spectrum = np.log1p(np.abs(fft_image))
    plt.imshow(spectrum, cmap='gray')
    plt.title('FFT Spectrum')
    plt.show()
    return fft_image

def notch_filter(shape, notch_coords, radius=10):
    mask = np.ones(shape, dtype=np.float32)
    for x, y in notch_coords:
        cv, cu = np.ogrid[:shape[0], :shape[1]]
        dist = np.sqrt((cu - y) ** 2 + (cv - x) ** 2)
        mask[dist <= radius] = 0
    return mask

def bandpass_filter(shape, inner_radius, outer_radius):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    cv, cu = np.ogrid[:rows, :cols]
    dist = np.sqrt((cu - ccol) ** 2 + (cv - crow) ** 2)
    mask[(dist >= inner_radius) & (dist <= outer_radius)] = 1
    return mask

def apply_filter(image, filter_mask):
    fft_image = np.fft.fftshift(np.fft.fft2(image))
    filtered_fft = fft_image * filter_mask
    filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))
    return filtered_image

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 NoiseRemoval.py <original_image_path> <noisy_image_path>")
        sys.exit(1)

    original_img = load_image(sys.argv[1])
    noisy1 = load_image(sys.argv[2])

    fft_noisy1 = show_fft(noisy1)

    notch_coords = [(100, 120), (140, 160)]
    notch_mask = notch_filter(noisy1.shape, notch_coords)
    filtered_notch1 = apply_filter(noisy1, notch_mask)

    band_mask = bandpass_filter(noisy1.shape, 0, 200)
    filtered_band1 = apply_filter(noisy1, band_mask)

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(original_img, cmap='gray', interpolation='nearest', vmax=255, vmin=0)
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(noisy1, cmap='gray', interpolation='nearest', vmax=255, vmin=0)
    plt.title('Noisy Image 1')

    plt.subplot(2, 2, 3)
    plt.imshow(filtered_notch1, cmap='gray', interpolation='nearest', vmax=255, vmin=0)
    plt.title('Notch Filtered')


    plt.subplot(2, 2, 4)
    plt.imshow(filtered_band1, cmap='gray', interpolation='nearest', vmax=255, vmin=0)
    plt.title('Bandpass Filtered')

    plt.tight_layout()
    plt.show()

