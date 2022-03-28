import math
from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import feature
from skimage.feature import peak_local_max
from skimage.io import imread
from skimage.transform import hough_line, hough_line_peaks, hough_circle, hough_circle_peaks

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

    # Thetas from [-90 to 90) (180 degrees) in radians
    theta_step = math.pi / (theta_res * 180)
    thetas = np.arange(-math.pi / 2, math.pi / 2, theta_step)

    thetas_is = np.arange(0, len(thetas))

    hough_space = np.zeros((len(rhos), len(thetas)))  # row x col

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x, y] == 0:
                continue

            rhos_is = (x * np.cos(thetas) + y * np.sin(thetas)).astype('int32') + rho_max
            hough_space[rhos_is, thetas_is] += 1

    return hough_space, thetas, rhos


def calc_y(x: int, theta: float, rho: float):
    """
    Calculates a line from its Normal Form
    Args:
        x: X in image space
        theta: X in parameter space, angle in radians
        rho: Y in parameter space,

    Returns:
        The y-value related to the input x value, pointing to a point in image space.
    """
    if theta == 0.0:
        return rho
    return (-math.cos(theta) / math.sin(theta)) * x + (rho / math.sin(theta))


def hough_lines(image, thetas, rhos, peaks):
    lines = []
    for rho_i, theta_i in peaks:  # row, col
        theta = thetas[theta_i]
        rho = int(rhos[rho_i])
        x1 = 0
        x2 = image.shape[0]
        y1 = calc_y(x1, theta, rho)
        y2 = calc_y(x2, theta, rho)
        lines.append(([x1, x2], [y1, y2]))
    return lines


def circle_hough_transform(image, sigma=1.4, low_threshold=0.8, high_threshold=0.9, radii=None):
    canny_image = feature.canny(image,
                                sigma=sigma,
                                low_threshold=low_threshold,
                                high_threshold=high_threshold,
                                use_quantiles=True)
    if radii is None:
        radii = np.arange(5, 200, 2)

    circles = hough_circle(canny_image, radius=radii)
    return circles, radii, canny_image


def hough_circles(image, radii, num_peaks=10):
    accums, cxs, cys, radii = hough_circle_peaks(image, radii, total_num_peaks=num_peaks)
    circles = []
    for cy, cx, radius in zip(cys, cxs, radii):
        circles.append((cx, cy, radius))
    return circles


def task2():
    image_name = "cross.png"
    image = imread(f"Week7/images/{image_name}", as_gray=True)

    def plot(hough_image, lines, peaks, thetas, rhos):
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        axs[0].set_xlim([0, image.shape[0] - 1])
        axs[0].set_ylim([image.shape[0] - 1, 0])
        axs[0].set_title(f"{image_name}")
        axs[0].imshow(image, cmap='gray', aspect="auto")
        colors = ['r', 'b', 'green', 'orange']
        for i, (line, peak) in enumerate(zip(lines, peaks)):
            axs[0].plot(line[0], line[1], c=colors[i])
            x, y = -(peak[1] - 90), peak[0] - rhos[-1]
            sc = axs[1].scatter(x, y, s=200, marker='s', edgecolors=colors[i])
            sc.set_facecolor("none")
        hack = axs[1].imshow(hough_image, cmap='inferno', aspect="auto")
        fig.colorbar(hack, ax=axs[1])
        axs[1].imshow(np.log(1 + hough_image),
                      extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]],
                      cmap='inferno', aspect="auto")
        axs[1].set_xlabel('theta')
        axs[1].set_ylabel('rho')
        axs[1].set_title("Hough Space")

    # Ours
    hough_image, thetas, rhos = line_hough_transform(image, 1, 1)
    peaks = peak_local_max(hough_image, num_peaks=2)
    lines = hough_lines(image, thetas, rhos, peaks)
    plot(hough_image, lines, peaks, thetas, rhos)

    # skimage
    hough_image, theta, distances = hough_line(image)
    accum, angles, dists = hough_line_peaks(hough_image, theta, distances, num_peaks=2)
    peaks = list(zip(*[[xs, ys] for a in np.unique(accum) for xs, ys in np.where(hough_image == a)]))
    lines = []
    for _, angle, dist in zip(*hough_line_peaks(hough_image, theta, distances)):
        y1 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y2 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
        lines.append(([0, image.shape[0]], [y1, y2]))
    plot(hough_image, lines, peaks, thetas, rhos)

    plt.show()


def task3():
    image_name = "coins.png"
    image = imread(f"Week7/images/{image_name}", as_gray=True)

    hough_space, radii, canny_image = circle_hough_transform(image)
    circles = hough_circles(hough_space, radii, num_peaks=10)

    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    axs[0][0].imshow(image, cmap='gray', aspect="auto")
    axs[0][0].set_title(f"{image_name}")

    axs[0][1].imshow(canny_image, cmap='gray', aspect="auto")
    axs[0][1].set_title("Canny coins")

    h_im = axs[1][0].imshow(hough_space[0], cmap='inferno', aspect="auto")
    axs[1][0].set_title(f"Hough Space, Radius {radii[0]}")
    fig.colorbar(h_im, ax=axs[1][0])

    axs[1][1].imshow(image, cmap='gray', aspect="auto")
    axs[1][1].set_title("Segmented coins")

    for center_x, center_y, radius in circles:
        circle = plt.Circle((center_x, center_y), radius, color='r')
        axs[1][1].add_patch(circle)
    plt.show()


if __name__ == '__main__':
    task3()
