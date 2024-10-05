import numpy as np
from libtiff import TIFF
from matplotlib import pyplot as plt

tifImg = TIFF.open('ImgSet1/imgset1/TMA2-v2.tif')
plt.imshow(TIFF.read_image(tifImg), interpolation='nearest', vmin=0, vmax=255)
plt.show()
tifImg.close()