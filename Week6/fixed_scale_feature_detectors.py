from itertools import product

import skimage
from matplotlib import pyplot as plt
from skimage import feature
from skimage.io import imread
from scipy.ndimage import median_filter, gaussian_filter


def task1():
    image = imread("Week6/images/hand.tiff", as_gray=True)

    sigmas = [0.1, 1, 5, 10]
    low_thresholds = [0.1, 0.1, 0.5, 0.5]
    high_thresholds = [0.6, 0.9, 0.6, 0.9]

    fig, axs = plt.subplots(4, 5, figsize=(16, 16))
    for a in axs:
        for ax in a:
            ax.set_yticks([])
            ax.set_xticks([])

    axs[0][0].set_title(f"Gaussian Original")
    for ax, (lt, ht) in zip(axs[0][1:], zip(low_thresholds, high_thresholds)):
        ax.set_title(f"Low T: {lt}, High T: {ht}")

    for ax, sigma in zip(axs[:, 0], sigmas):
        ax.set_ylabel(f"Sigma: {sigma}                        ", rotation=0, size='large')

    axs[0][0].imshow(image, cmap="gray")
    for i, sigma in enumerate(sigmas):
        gauss_image = gaussian_filter(image, sigma=sigma)
        axs[i][0].imshow(gauss_image, cmap="gray", aspect='auto')
        for j, (low_threshold, high_threshold) in enumerate(zip(low_thresholds, high_thresholds)):
            canny_image = feature.canny(image, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold, use_quantiles=True)

            axs[i][j+1].imshow(canny_image, cmap="gray", aspect='auto')

    plt.show()

if __name__ == '__main__':
    task1()

