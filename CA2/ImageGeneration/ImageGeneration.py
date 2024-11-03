import numpy as np
import matplotlib.pyplot as plt

def compute_dft(image):
    dft = np.fft.fft2(image)
    dft_shifted = np.fft.fftshift(dft)
    magnitude_spectrum = np.log(np.abs(dft_shifted) + 1)
    return magnitude_spectrum

def generate_sparse_points(size=256, num_points=5):
    pattern = np.zeros((size, size))
    points = np.random.randint(0, size, (num_points, 2))

    for point in points:
        pattern[point[0], point[1]] = 1
    return pattern

def generate_sinusoidal_pattern(size=256, freq=20, angle=0):
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    
    rotated_X = X * np.cos(angle) + Y * np.sin(angle)
    return np.cos(2 * np.pi * freq * rotated_X / size)

def generate_grid_pattern(size=256, freq_x=10, freq_y=10):
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    
    pattern = (np.cos(2 * np.pi * freq_x * X / size) + 
               np.cos(2 * np.pi * freq_y * Y / size))
    
    # Normalize
    return (pattern - pattern.min()) / (pattern.max() - pattern.min())

def generate_pyramid_shape(size=256):
    pattern = np.zeros((size, size))
    height = size // 2
    width = size // 4
    center_x = size // 2
    center_y = size // 2

    for y in range(height):
        for x in range(width):
            if (y >= height // 2 and x < width - 1 - y // 2):
                pattern[center_y - height // 2 + y, center_x - width // 2 + x] = 1
            
            if (y >= height // 2 and x < width - 1 - y // 2):
                pattern[center_y - height // 2 + y, center_x + width // 2 - x] = 1
            
            if (y == height // 2 and x < width):
                pattern[center_y - height // 2 + y, center_x - width // 2 + x] = 1

    return pattern

def display_pattern_and_dft(pattern, pattern_name):
    dft_spectrum = compute_dft(pattern)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(pattern, cmap='gray')
    plt.title(f"{pattern_name}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(dft_spectrum, cmap='gray')
    plt.title(f"DFT Spectrum of {pattern_name}")
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    # Random points
    sparse_points_pattern = generate_sparse_points(size=256, num_points=10)
    display_pattern_and_dft(sparse_points_pattern, "Sparse Points")

    # Sin (Horizontal)
    sinusoidal_pattern_h = generate_sinusoidal_pattern(size=256, freq=15, angle=0)
    display_pattern_and_dft(sinusoidal_pattern_h, "High-Frequency Sinusoidal Waves (Vertical)")

    # Sin (Vertical)
    sinusoidal_pattern_v = generate_sinusoidal_pattern(size=256, freq=15, angle=np.pi / 2)
    display_pattern_and_dft(sinusoidal_pattern_v, "High-Frequency Sinusoidal Waves (Horizontal)")

    # Grid
    grid_pattern = generate_grid_pattern(size=256, freq_x=10, freq_y=10)
    display_pattern_and_dft(grid_pattern, "Grid Pattern")

    # Pyramid
    pyramid_pattern = generate_pyramid_shape(size=256)
    display_pattern_and_dft(pyramid_pattern, "Pyramid pattern")
