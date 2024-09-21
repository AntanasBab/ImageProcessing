from matplotlib import pyplot as plt
from libtiff import TIFF

tifImg = TIFF.open('ImgSet1/imgset1/Kidney2_RGB2_20x.svs')
img = TIFF.read_image(tifImg)
plt.imshow(img, interpolation='nearest')
plt.show()
tifImg.close()