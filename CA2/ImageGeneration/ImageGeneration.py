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
    return (pattern - pattern.min()) / (pattern.max() - pattern.min()) 

def generate_circular_pattern(size=256, num_circles=10):
    pattern = np.zeros((size, size))
    center = size // 2
    
    for i in range(1, num_circles + 1):
        radius = i * (size // (2 * num_circles))
        y, x = np.ogrid[-center:size - center, -center:size - center]
        mask = x**2 + y**2 <= radius**2
        
        pattern[mask] = i / num_circles 
    
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
    sparse_points_pattern = generate_sparse_points(size=256, num_points=5)
    display_pattern_and_dft(sparse_points_pattern, "Sparse Points")

    # Sin (Horizontal)
    sinusoidal_pattern_h = generate_sinusoidal_pattern(size=256, freq=15, angle=0)
    display_pattern_and_dft(sinusoidal_pattern_h, "High-Frequency Sinusoidal Waves (Horizontal)")

    # Sin (Vertical)
    sinusoidal_pattern_v = generate_sinusoidal_pattern(size=256, freq=15, angle=np.pi / 2)
    display_pattern_and_dft(sinusoidal_pattern_v, "High-Frequency Sinusoidal Waves (Vertical)")

    # Grid
    grid_pattern = generate_grid_pattern(size=256, freq_x=10, freq_y=10)
    display_pattern_and_dft(grid_pattern, "Grid Pattern")

    # Circle
    circular_pattern = generate_circular_pattern(size=256, num_circles=10)
    display_pattern_and_dft(circular_pattern, "Circular Pattern")
