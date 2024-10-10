import numpy as np
from libtiff import TIFF
from matplotlib import pyplot as plt
import sys

def readImgSubRegion(tiffImg, tile_x1, tile_y1, tile_x2, tile_y2): 
    tile_width = tiffImg.GetField('TileWidth')
    tile_length = tiffImg.GetField('TileLength')
    img_width = tiffImg.GetField('ImageWidth')
    img_length = tiffImg.GetField('ImageLength')

    max_tiles_x = np.ceil(img_width / tile_width).astype(int) 
    max_tiles_y = np.ceil(img_length / tile_length).astype(int) 

    if tile_x2 > max_tiles_x or tile_y2 > max_tiles_y:
        print(f"Error: Requested region exceeds available tiles. Maximum X tiles: {max_tiles_x}, Maximum Y tiles: {max_tiles_y}.")
        return None

    res = []
    for tile_row in range(tile_y1, tile_y2):
        row_tiles = []
        for tile_col in range(tile_x1, tile_x2):
            tile = TIFF.read_one_tile(tiffImg, tile_col * tile_length, tile_row * tile_width)
            row_tiles.append(tile)
        res.append(np.concatenate(row_tiles, axis=1))
    result = np.concatenate(res, axis=0)
    return result

if len(sys.argv) != 2:
    print("Usage: python3 displayTiledImgRegions.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

tifImg = TIFF.open(image_path)
img = readImgSubRegion(tifImg, 0, 0, 5, 11)
plt.imshow(img, interpolation='nearest', vmin=0, vmax=255)
plt.show()
tifImg.close()
