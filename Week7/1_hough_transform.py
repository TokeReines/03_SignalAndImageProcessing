import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import feature
from skimage.feature import peak_local_max
from skimage.io import imread


def line_hough_transform(image, theta_res: int = 1, rho_res: int = 1):
    """
    Image space dots are lines in parameterspace. Dots can be edges in a binary image constructed by edge detection.

    Cartesian line: y = a * x + b
        a = gradient or slope
        b = y-translation
    Where (x, y) defines the dot in image space

    Parameter space line:  b = -a * x + y
    Where (a, b) defines the line in parameterspace.

    Since -inf <= a <= inf (we cannot detect vertical lines) we use a discretised version of the normal line:
        p = x * cos(theta) - y * sin(theta)
    Where 0 <= theta <= pi, and p <= image diagonal

    Args:
        rho_res: Steps pr rho (2x diagonal)
        theta_res: Steps per degree of theta
        image: Binary image with edges from edge detection. As a numpy array

    Returns:
        Hough Transform image with sinusoidal curves to infer lines in the input image from.
    """

    rho_max = int(math.hypot(*image.shape))
    num_rhos = rho_max * rho_res * 2
    rhos = np.linspace(-rho_max, rho_max, num_rhos)

    # Thetas from -90 to 90 (180 degrees) in radians
    num_thetas = theta_res * 180
    thetas = np.linspace(-math.pi / 2, math.pi / 2, num_thetas)
    thetas_is = np.arange(0, len(thetas))

    hough_space = np.zeros((len(rhos), len(thetas)))

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x, y] == 0:
                continue

            rhos_is = (x * np.cos(thetas) + y * np.sin(thetas)).astype('int32') + rho_max
            hough_space[rhos_is, thetas_is] += 1

    return hough_space, thetas, rhos


def hough_lines(image, hough_image, thetas, rhos, num_peaks=4):
    def calc_y(x, theta, rho):
        if theta == 0.0:
            return rho
        return (-math.cos(theta) / math.sin(theta)) * x + (rho / math.sin(theta))

    peaks = peak_local_max(hough_image, num_peaks=num_peaks)
    lines = []
    for rho_i, theta_i in peaks:  # row, col
        theta = thetas[theta_i]
        rho = rhos[rho_i]
        x1 = 0
        x2 = image.shape[0]
        y1 = calc_y(x1, theta, rho)
        y2 = calc_y(x2, theta, rho)
        lines.append(([x1, x2], [y1, y2]))
    return lines, peaks


if __name__ == '__main__':
    image = imread("Week7/images/cross.png", as_gray=True)

    hough_image, thetas, rhos = line_hough_transform(image, 1, 1)
    lines, peaks = hough_lines(image, hough_image, thetas, rhos, num_peaks=2)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].set_xlim([0, image.shape[0] - 1])
    axs[0].set_ylim([image.shape[0] - 1, 0])
    axs[0].imshow(image, cmap='gray')
    colors = ['r', 'b', 'green', 'orange']
    for i, (line, peak) in enumerate(zip(lines, peaks)):
        axs[0].plot(line[0], line[1], c=colors[i])
        sc = axs[1].scatter(peak[1], peak[0], s=200, marker='s', edgecolors=colors[i])
        sc.set_facecolor("none")
    axs[1].imshow(hough_image, cmap='inferno')
    axs[1].set_xlabel('theta')
    axs[1].set_ylabel('rho')
    plt.show()
