from matplotlib import pyplot as plt
from libtiff import TIFF
from imgInfo import isImgTiled
import numpy as np

def readImgSubRegion(tiffImg, x1, y1, x2, y2):
    if not isImgTiled(tiffImg):
        print('Image is not tiled, aborting')
        return

    tile_width = tiffImg.GetField('TileWidth')
    tile_length = tiffImg.GetField('TileLength')
    image_width = tiffImg.GetField('ImageWidth')
    image_height = tiffImg.GetField('ImageLength')
   
    # Ensure coordinates are within bounds
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    tile_x1 = x1 // tile_width
    tile_y1 = y1 // tile_length
    tile_x2 = x2 // tile_width
    tile_y2 = y2 // tile_length

    # Calculate the size of the output array
    region_height = y2 - y1 + 1
    region_width = x2 - x1 + 1
    samples_per_pixel = tiffImg.GetField('SamplesPerPixel') or 1
    
    # Initialize the subregion array
    subregion = np.zeros((region_height, region_width, samples_per_pixel), dtype=np.uint8)

    for tile_row in range(tile_y1, tile_y2 + 1):
        for tile_col in range(tile_x1, tile_x2 + 1):
            tile = TIFF.read_one_tile(tiffImg, tile_row, tile_col)  # Adjust according to your ReadTile method
            if tile is not None:
                # Calculate tile boundaries
                tile_x_start = tile_col * tile_width
                tile_y_start = tile_row * tile_length

                # Calculate overlap with requested subregion
                x_start = max(x1, tile_x_start) - tile_x_start
                y_start = max(y1, tile_y_start) - tile_y_start
                x_end = min(x2, tile_x_start + tile_width) - tile_x_start
                y_end = min(y2, tile_y_start + tile_length) - tile_y_start

                # Insert the tile data into the appropriate location in the subregion
                subregion[y_start:y_end, x_start:x_end, :] = tile[y_start:y_end, x_start:x_end, :]

    return subregion

# Example usage
tifImg = TIFF.open('ImgSet1/imgset1/Kidney1.tif')
img = readImgSubRegion(tifImg, 10, 20, 10, 40)
plt.imshow(img, interpolation='nearest')
plt.show()
tifImg.close()