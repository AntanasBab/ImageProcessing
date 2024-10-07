import sys
import numpy as np
from libtiff import TIFF
from matplotlib import pyplot as plt

class ImageConverter:
    def __init__(self, image_8bit=None, image_float=None):
        self.image_8bit = image_8bit   
        self.image_float = image_float 

    def load_image(self, image_path):
        tifImg = TIFF.open(image_path)
        image = TIFF.read_image(tifImg)
        tifImg.close()
        return image


    def convert_8bit_to_float(self, image_8bit):
        if image_8bit is None:
            raise ValueError("8-bit image is not provided.")
      
        self.image_float = image_8bit.astype(np.float32)
        return self.image_float
    

    def convert_float_to_8bit(self, image_float):
        if image_float is None:
            raise ValueError("Float image is not provided.")

        image_clipped = np.clip(image_float, 0, 255)
        self.image_8bit = image_clipped.astype(np.uint8)
        return self.image_8bit

def main(image_path):
    converter = ImageConverter()

    image_8bit = converter.load_image(image_path)
    print("Original 8-bit Image:")
    print(image_8bit)

    image_float = converter.convert_8bit_to_float(image_8bit)
    print("\nConverted Float Image:")
    print(image_float)

    image_8bit_clipped = converter.convert_float_to_8bit(image_float)
    print("\nConverted Back to 8-bit Image:")
    print(image_8bit_clipped)


image_path = sys.argv[1]
main(image_path)
