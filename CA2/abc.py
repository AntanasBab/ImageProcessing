import numpy as np
import matplotlib.pyplot as plt
from libtiff import TIFF

# Step 1: Load and display the input image (M x N)
tiff = TIFF.open('/home/antbab/ImageProcessing/CA2/imgset5/Fig0429(a)(blown_ic_crop).tif', mode='r')
image = tiff.read_image()
M, N = image.shape

plt.figure(figsize=(10, 8))
plt.subplot(2, 3, 1)
plt.title("1. Original Image (MxN)")
plt.imshow(image, cmap='gray')

# Step 2: Pad the image to 2M x 2N
M, N = image.shape
padded_image = np.zeros((2 * M, 2 * N), dtype=np.complex128) 

for i in range(M): 
    for j in range(N): 
        padded_image[i, j] = image[i, j]  

plt.subplot(2, 3, 2)
plt.title("2. Padded Image (2Mx2N)")
plt.imshow(np.abs(padded_image), cmap='gray')

# Step 3: Apply the shift manually using (-1)^(x+y)
x = np.arange(2 * M)
y = np.arange(2 * N)
X, Y = np.meshgrid(x, y, indexing='ij')
shifted_image = padded_image * ((-1) ** (X + Y))

# Dimming the shifted image (e.g., reduce brightness by 50%)
dimming_factor = 0.5
shifted_image_dimmer = shifted_image * dimming_factor

# Normalize the shifted image for display
shifted_image_display = np.abs(shifted_image_dimmer)

plt.subplot(2, 3, 3)
plt.title("3. Shifted Image for Periodicity")
plt.imshow(shifted_image_display, cmap='gray', interpolation='nearest', vmax=255, vmin=0)

# Step 4: Compute the DFT of the shifted image
dft_image = np.fft.fft2(shifted_image)

# Display the magnitude of the DFT for visualization
magnitude_spectrum = np.log(np.abs(dft_image) + 1)

plt.subplot(2, 3, 4)
plt.title("4. DFT Magnitude Spectrum")
plt.imshow(magnitude_spectrum, cmap='gray')

# Step 5: Compute the Inverse DFT
idft_image = np.fft.ifft2(dft_image)

# Apply the shift back manually to return to the original alignment
shifted_idft_image = idft_image * ((-1) ** (X + Y))

plt.subplot(2, 3, 5)
plt.title("5. Inverse DFT (Shifted Back)")
plt.imshow(np.abs(shifted_idft_image), cmap='gray')

# Step 6: Crop to the original image size (upper-left quadrant)
final_image = np.abs(shifted_idft_image[:M, :N])

plt.subplot(2, 3, 6)
plt.title("6. Cropped Final Image")
plt.imshow(final_image, cmap='gray')

plt.tight_layout()
plt.show()
