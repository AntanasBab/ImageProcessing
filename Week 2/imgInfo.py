from libtiff import TIFF
import os

def getCMPStr(intVal):
    match intVal:
        case 1:
            return 'Compression Scheme: none'
        case 5:
            return 'Compression Scheme: LZW'
        case 7:
            return 'Compression Scheme: JPEG'
    
    return 'Unknown'

def getPhotometricStr(intVal):
    photometricValue = {
        0: "WhiteIsZero (For bilevel and grayscale images, 0 is imaged as white)",
        1: "BlackIsZero (For bilevel and grayscale images, 0 is imaged as black)",
        2: "RGB (RGB color model)",
        3: "Palette Color (Color-mapped image)",
        4: "Transparency Mask (Transparency mask used)",
        5: "CMYK (Separation color)",
        6: "YCbCr (Luminance and chrominance color model)",
        8: "CIELab (CIELAB color space)"
    }

    return photometricValue.get(intVal, "Unknown Photometric Interpretation")

def getImgPlane(intVal):
    return 'single image plane' if intVal == 1 else 'seperate image plane'

def isImgTiled(tifImg):
    tile_width = tifImg.GetField('TileWidth')
    tile_length = tifImg.GetField('TileLength')
    rows_per_strip = tifImg.GetField('RowsPerStrip')

    if tile_width and tile_length:
        return True
    elif rows_per_strip:
        return False
    else:
        raise Exception(f"Image does not contain tiling or strip information.")

def ImgTiffinfo(tifImg):
    print('Subfile Type: ' + str(tifImg.GetField('SUBFILETYPE')))
    print('Image Width: ' + str(tifImg.GetField('IMAGEWIDTH')) + ' Image Length: ' + str(tifImg.GetField('IMAGELENGTH')) + ' Image Depth: ' + str(tifImg.GetField('IMAGEDEPTH')))

    print('-------------------')
    tile_width = tifImg.GetField('TileWidth')
    tile_length = tifImg.GetField('TileLength')
    rows_per_strip = tifImg.GetField('RowsPerStrip')
    isTiled = isImgTiled(tifImg)

    if isTiled:
        print(f"Image is tiled with TileWidth={tile_width}, TileLength={tile_length}.")
    else:
        print(f"Image is striped with RowsPerStrip={rows_per_strip}.")

    print('-------------------')
    print('Bits/Sample: ' + str(tifImg.GetField('BITSPERSAMPLE')))
    print(getCMPStr(tifImg.GetField('COMPRESSION')))
    print('Photometric Interpretation: ' + getPhotometricStr(tifImg.GetField('PHOTOMETRIC')))
    print('Samples/Pixel: ' + str(tifImg.GetField('SAMPLESPERPIXEL')))
    print('Planar Configuration: ' + getImgPlane(tifImg.GetField('PLANARCONFIG')))
    print('Image Description: ' + str(tifImg.GetField('IMAGEDESCRIPTION')))
    print('\n')


def dirImgInfo(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            print(file)
            tiff_file = os.path.join(root, file)
            tifImg = TIFF.open(tiff_file)
            ImgTiffinfo(tifImg)
            tifImg.close()

# dirImgInfo('./ImgSet1')