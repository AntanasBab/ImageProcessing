import sys
import numpy as np
from libtiff import TIFF
from matplotlib import pyplot as plt

def load_image(image_path):
    tifImg = TIFF.open(image_path)
    image = TIFF.read_image(tifImg)
    tifImg.close()
    return image

def convert_8bit_to_float(image_8bit):
    if image_8bit is None:
        raise ValueError("8-bit image is not provided.")

    return image_8bit.astype(np.float32)


def convert_float_to_8bit(image_float):
    if image_float is None:
        raise ValueError("Float image is not provided.")

    image_clipped = np.clip(image_float, 0, 255)
    return image_clipped.astype(np.uint8)


def main(image_path):
    image_8bit = load_image(image_path)
    print("Original 8-bit Image:")
    print(image_8bit)

    image_float = convert_8bit_to_float(image_8bit)
    print("\nConverted Float Image:")
    print(image_float)


if len(sys.argv) != 2:
    print("Usage: python imageLoading.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
main(image_path)
