from libtiff import TIFF
import numpy as np

def load_tif_image(filename):
    tif = TIFF.open(filename, mode='r')
    image = tif.read_image()
    return np.array(image, dtype=np.float32)