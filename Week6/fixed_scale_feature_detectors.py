from itertools import product

import skimage
from matplotlib import pyplot as plt
from skimage import feature
from skimage.io import imread
from scipy.ndimage import median_filter, gaussian_filter


def task1():
    fig, axs = plt.subplots(4, 5, figsize=(16, 16))
    for a in axs:
        for ax in a:
            ax.set_yticks([])
            ax.set_xticks([])

    sigmas = [0.1, 1, 5, 10]
    low_thresholds = [0.1, 0.1, 0.5, 0.5]
    high_thresholds = [0.6, 0.9, 0.6, 0.9]
    axs[0][0].set_title(f"Gaussian Original")
    for ax, (lt, ht) in zip(axs[0][1:], zip(low_thresholds, high_thresholds)):
        ax.set_title(f"Low T: {lt}, High T: {ht}")

    for ax, sigma in zip(axs[:, 0], sigmas):
        ax.set_ylabel(f"Sigma: {sigma}                        ", rotation=0, size='large')

    image = imread("Week6/images/hand.tiff", as_gray=True)
    axs[0][0].imshow(image, cmap="gray")
    for i, sigma in enumerate(sigmas):
        gauss_image = gaussian_filter(image, sigma=sigma)
        axs[i][0].imshow(gauss_image, cmap="gray", aspect='auto')
        for j, (low_threshold, high_threshold) in enumerate(zip(low_thresholds, high_thresholds)):
            canny_image = feature.canny(image, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold, use_quantiles=True)

            axs[i][j+1].imshow(canny_image, cmap="gray", aspect='auto')

    plt.show()


def task2():
    fig, axs = plt.subplots(4, 5, figsize=(16, 16))
    for a in axs:
        for ax in a:
            ax.set_yticks([])
            ax.set_xticks([])

    sigmas = [0.1, 1, 5, 10]
    ks = [0.001, 0.01, 0.1, 0.2]

    axs[0][0].set_title(f"Gaussian Original")
    for ax, sigma in zip(axs[:, 0], sigmas):
        ax.set_ylabel(f"Sigma: {sigma}                        ", rotation=0, size='large')
    for ax, k in zip(axs[0][1:], ks):
        ax.set_title(f"K: {k}")

    image = imread("Week6/images/modelhouses.png", as_gray=True)
    for i, sigma in enumerate(sigmas):
        gauss_image = gaussian_filter(image, sigma=sigma)
        axs[i][0].imshow(gauss_image, cmap="gray", aspect='auto')
        for j, k in enumerate(ks):
            corners = feature.corner_harris(image, k=k, sigma=sigma)
            axs[i][j+1].imshow(corners, cmap="gray", aspect='auto')

    fig, axs = plt.subplots(4, 5, figsize=(16, 16))
    for a in axs:
        for ax in a:
            ax.set_yticks([])
            ax.set_xticks([])
    epss = [1e-12, 1e-6, 1, 5]

    axs[0][0].set_title(f"Gaussian Original")
    for ax, sigma in zip(axs[:, 0], sigmas):
        ax.set_ylabel(f"Sigma: {sigma}                        ", rotation=0, size='large')
    for ax, eps in zip(axs[0][1:], epss):
        ax.set_title(f"Eps: {eps}")

    for i, sigma in enumerate(sigmas):
        gauss_image = gaussian_filter(image, sigma=sigma)
        axs[i][0].imshow(gauss_image, cmap="gray", aspect='auto')
        for j, eps in enumerate(epss):
            corners = feature.corner_harris(image, method='eps', eps=eps, sigma=sigma)
            axs[i][j+1].imshow(corners, cmap="gray", aspect='auto')

    plt.show()


def task3():
    image = imread("Week6/images/modelhouses.png", as_gray=True)

    k = 0.001
    sigma = 3
    min_distance = 1
    threshold_rel = 0.001
    response = feature.corner_harris(image, k=k, sigma=sigma)
    corners = feature.corner_peaks(response, min_distance=min_distance, threshold_rel=threshold_rel)
    fig, ax = plt.subplots(1, 1, figsize=(24, 24))

    ax.set_yticks([])
    ax.set_xticks([])

    ax.imshow(image, cmap="gray", aspect='auto')
    ax.scatter(corners[:,1], corners[:,0], color='r', marker='D', s=5)
    ax.set_title(f"Corners, k={k}, sigma={sigma}, min_distance={min_distance}, threshold_rel={threshold_rel}")
    # axs[1].imshow(response, cmap="gray", aspect='auto')
    # axs[1].set_title(f"Response image")
    #
    # gauss_image = gaussian_filter(image, sigma=sigma)
    # axs[2].imshow(gauss_image, cmap="gray", aspect='auto')
    # axs[2].set_title(f"Gaussian blurred")
    plt.show()
    a = 2

if __name__ == '__main__':
    task3()

