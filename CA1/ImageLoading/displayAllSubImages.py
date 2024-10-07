from libtiff import TIFF
import matplotlib.pyplot as plt

def viewSubImages(file_path):
    tifImg = TIFF.open(file_path)
    directories = list(TIFF.iter_images(tifImg))
    num_images = len(directories)

    fig, axs = plt.subplots(1, num_images, figsize=(15, 15))
    if num_images == 1:
        axs = [axs]
    
    for idx, image in enumerate(directories):
        axs[idx].imshow(image, interpolation='nearest', vmin=0, vmax=255)
        axs[idx].set_title(f'Subimage {idx+1}')
        axs[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

viewSubImages('ImgSet1/imgset1/Kidney2_RGB2_20x.svs')
