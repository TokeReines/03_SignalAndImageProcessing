from scipy.ndimage import binary_hit_or_miss
from skimage.io import imread
from skimage import morphology
import matplotlib.pyplot as plt
import numpy as np

def task11():
    image = np.array(imread("Week5/images/cells_binary.png", as_gray=True)).astype('uint8')
    disk = morphology.disk(2)
    opened_image = morphology.opening(image, disk)
    closed_image = morphology.closing(image, disk)

    fig, axs = plt.subplots(3, 1, figsize=(8, 16))

    im =axs[0].imshow(image, cmap="gray", aspect='auto')
    axs[0].set_title("Original")

    axs[1].imshow(opened_image, cmap="gray", aspect='auto')
    axs[1].set_title("Opened")

    axs[2].imshow(closed_image, cmap="gray", aspect='auto')
    axs[2].set_title("Closed")

    fig.colorbar(im, ax=axs[2])
    plt.show()


if __name__ == '__main__':
    image = np.array(imread("Week5/images/blobs_inv.png", as_gray=True)).astype('uint8')
    image[image > 0] = 1
    vertical_line = np.array([[1],
                              [1],
                              [1],
                              [1],
                              [1]])
    disk = np.array([[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]])
    disk = morphology.disk(2)
    corner = np.array([[0, 0, 0],
                       [1, 1, 0],
                       [1, 1, 0]])

    SEs = [vertical_line, disk, corner]

    fig, axs = plt.subplots(4, len(SEs), figsize=(8, 16))

    for i, SE in enumerate(SEs):
        white_tophat = morphology.white_tophat(image, SE)
        black_tophat = morphology.black_tophat(image, SE)
        hit_or_miss = binary_hit_or_miss(image, structure1=SE).astype('uint8')

        axs[0][i].imshow(SE, cmap="gray", aspect='auto', vmin=0, vmax=1)
        axs[0][i].set_title("SE")
        axs[1][i].imshow(white_tophat, cmap="gray", aspect='auto')
        axs[1][i].set_title("White tophat")
        axs[2][i].imshow(black_tophat, cmap="gray", aspect='auto')
        axs[2][i].set_title("Black tophat")
        axs[3][i].imshow(hit_or_miss, cmap="gray", aspect='auto')
        axs[3][i].set_title("Hit or miss")
        #black_tophat = morphology.binary.(image, SE)
        #hit_or_miss()
    plt.show()