import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift

def plot_image_and_spectrum(image, title):
    # Compute Fourier Transform and shift the zero frequency to the center
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)

    # Display the image and its Fourier spectrum
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title(f"{title}")
    axs[0].axis("off")
    
    axs[1].imshow(magnitude_spectrum, cmap='gray')
    axs[1].set_title(f"Fourier Spectrum of {title}")
    axs[1].axis("off")
    plt.show()

# 1. Constant Image
constant_image = np.ones((256, 256))
plot_image_and_spectrum(constant_image, "Constant Image")

# 2. Single Frequency Pattern (Horizontal Sine Wave)
x = np.linspace(0, 2 * np.pi, 256)
single_frequency_image = np.sin(5 * x)  # Horizontal sine pattern
single_frequency_image = np.tile(single_frequency_image, (256, 1))
plot_image_and_spectrum(single_frequency_image, "Single Frequency Pattern")

# 3. Grid Pattern (Checkerboard)
grid_image = np.indices((256, 256)).sum(axis=0) % 2  # Simple checkerboard pattern
plot_image_and_spectrum(grid_image, "Grid Pattern")

# 4. Random Noise Image
random_noise_image = np.random.rand(256, 256)
plot_image_and_spectrum(random_noise_image, "Random Noise")

# 5. Radial Gradient
y, x = np.ogrid[:256, :256]
center = (128, 128)
radial_gradient_image = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
radial_gradient_image = (radial_gradient_image / radial_gradient_image.max()) * 255
plot_image_and_spectrum(radial_gradient_image, "Radial Gradient")
