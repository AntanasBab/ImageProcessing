import sys
from libtiff import TIFF
from matplotlib import pyplot as plt

if len(sys.argv) != 2:
    print("Usage: python3 displayStripedImg.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
tifImg = TIFF.open(image_path)

plt.imshow(TIFF.read_image(tifImg), interpolation='nearest', vmin=0, vmax=255)
plt.show()

tifImg.close()
