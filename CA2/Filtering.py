from FourierTransform import *
import sys
import numpy as np
import libtiff
import matplotlib.pyplot as plt

def ideal_low_pass_filter(shape, D0):
    P, Q = shape
    # Centers
    cP, cQ = P // 2, Q // 2
    mask = np.zeros((P, Q), dtype=np.float32)
    
    for i in range(P):
        for j in range(Q):
            distance = (i - cP) ** 2 + (j - cQ) ** 2
            if distance <= D0 ** 2:
                mask[i, j] = 1.0
                
    return mask

def apply_filter(image, cutoff):
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    lp_filter = ideal_low_pass_filter(image.shape, cutoff)
    filtered_transform = f_transform_shifted * lp_filter
    
    filtered_image = np.fft.ifftshift(filtered_transform)
    img_back = np.fft.ifft2(filtered_image)
    img_back = np.abs(img_back)
    
    return img_back

if len(sys.argv) != 2:
    print("Usage: python FourierTransform.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
process_image(image_path)
