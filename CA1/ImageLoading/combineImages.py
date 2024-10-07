from libtiff import TIFF
import matplotlib.pyplot as plt
import numpy as np

tifImg1 = TIFF.open('ImgSet1/imgset1/Region_001_FOV_00041_Acridine_Or_Gray.tif')
tifImg2 = TIFF.open('ImgSet1/imgset1/Region_001_FOV_00041_DAPI_Gray.tif')
tifImg3 = TIFF.open('ImgSet1/imgset1/Region_001_FOV_00041_FITC_Gray.tif')

img1 = tifImg1.read_image()
img2 = tifImg2.read_image()
img3 = tifImg3.read_image()

combined_img = np.stack([img1, img2, img3], axis=-1)

plt.imshow(combined_img, interpolation='nearest', vmin=0, vmax=255) 
plt.axis('off')
plt.show()

TIFF.write_image(TIFF.open('combined.tif', mode='w'), combined_img, 'lzw',True)

tifImg1.close()
tifImg2.close()
tifImg3.close()
