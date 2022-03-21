import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
def G(size, sigma):
    # https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    x = (size[0] - ((size[0] -1) / 2))
    gauss = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * np.square(x) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def task3_1(sigma=0.1):

    def _soft_edge(y, x, sigma=2):
        x = (x - ((len(x) - 1) / 2))
        left = 1 / (np.sqrt(2 * np.pi * sigma ** 2))
        right = np.exp(-((x ** 2) / (2 * sigma ** 2)))
        return left * right

    size = (16, 16)
    points = np.fromfunction(_soft_edge, size, dtype=float)
    img = np.cumsum(points, axis=1)
    gaussian_kernel = G(size, 2)
    img_scale = convolve2d(img, gaussian_kernel)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].imshow(img, cmap='gray')
    axs[1].imshow(img_scale, cmap='gray')

if __name__ == "__main__":
    task3_1()
    plt.show()