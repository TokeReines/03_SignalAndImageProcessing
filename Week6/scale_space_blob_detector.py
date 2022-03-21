

# Assignment 6


# Packages

# Numpy, Matplotlib and miscellaneous
import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis, imshow, show, title, colorbar, figure
import math

# Scikit-image.
from skimage import img_as_ubyte, img_as_float
from skimage.io import imread

# Scipy.
from scipy import stats
import scipy.signal
from scipy.signal.windows import gaussian
from scipy.signal import convolve2d, fftconvolve
from scipy.ndimage import median_filter, gaussian_filter, convolve
from scipy import interpolate
from scipy import ndimage
from scipy.ndimage import shift
from skimage.feature import peak_local_max


# Scale-space blob detector


def Exercise_2_1():
    def G(size, sigma):
        # https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
        ax = np.linspace(-(size - 1) // 2, (size - 1) // 2, size)
        gauss = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)
        return kernel / np.sum(kernel)

    # Code snippet.
    L = 15
    s1 = G(L, 1)
    s2 = G(L, 2)
    s3 = G(L, sigma=np.sqrt(1**2 + 2**2))
    s_convo = fftconvolve(s2, s2, mode="same")
    s_diff = np.clip(s3 - s2, 0, 1)

    # Plotting
    fig, axs = plt.subplots(1, 5, figsize=(15, 6), facecolor='w', edgecolor='k')
    axs = axs.ravel()

    axs[0].imshow(s1, cmap='gray')
    axs[0].set_title("$G(x,y,\sigma)$")
    # axs[0].axis("off")

    axs[1].imshow(s2, cmap='gray')
    axs[1].set_title("$G(x,y,\\tau$)")
    axs[1].axis("off")

    axs[2].imshow(s3, cmap='gray')
    axs[2].set_title("$G(x,y,\sqrt{\sigma^{2} + \\tau^{2}})$")
    axs[2].axis("off")

    axs[3].imshow(s_convo, cmap='gray')
    axs[3].set_title("$G(x,y,\sigma)*G(x,y,\\tau)$")
    axs[3].axis("off")

    axs[4].imshow(s_diff, cmap='gray')
    axs[4].set_title("$G(x,y,\sqrt{\sigma^{2} + \\tau^{2}}) - G(x,y,\\tau)$")
    axs[4].axis("off")

    show()


def Exercise_2_3():
    def H(tau, sigma=1):
        return (-tau**2) / (np.pi * (sigma**2 + tau**2)**2)

    t = np.linspace(-0.5, 3, 100)  # Only positive values of tau.
    sigmas = [1, 1.2, 1.4, 1.6, 1.8, 2]
    Hs = [H(t, sigma=s) for s in sigmas]
    Ps = [H(s, s) for s in sigmas]

    # Plotting.
    plt.plot(t, Hs[0], 'r', label="$\sigma = 1.0$")
    plt.plot([sigmas[0]], [Ps[0]], marker='o', markersize=3, color="black")
    plt.plot(t, Hs[1], 'b', label="$\sigma = 1.2$")
    plt.plot([sigmas[1]], [Ps[1]], marker='o', markersize=3, color="black")
    plt.plot(t, Hs[2], 'g', label="$\sigma = 1.4$")
    plt.plot([sigmas[2]], [Ps[2]], marker='o', markersize=3, color="black")
    plt.plot(t, Hs[3], 'orange', label="$\sigma = 1.6$")
    plt.plot([sigmas[3]], [Ps[3]], marker='o', markersize=3, color="black")
    plt.plot(t, Hs[4], 'lime', label="$\sigma = 1.8$")
    plt.plot([sigmas[4]], [Ps[4]], marker='o', markersize=3, color="black")
    plt.plot(t, Hs[5], 'purple', label="$\sigma = 2.0$")
    plt.plot([sigmas[5]], [Ps[5]], marker='o', markersize=3, color="black")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.xticks(np.arange(0, 3, step=0.4))
    plt.xlabel("$\\tau$")
    plt.ylabel("$H(0,0,\\tau)$")
    plt.tight_layout()
    plt.show()

def Exercise_2_4():
    def G(size, sigma):
        #size = abs(int(2 * sigma))
        # https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
        ax = np.linspace(-(size - 1) // 2, (size - 1) // 2, size)
        gauss = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)
        return kernel / np.sum(kernel)

    image = imread("Week6/images/sunflower.tiff", as_gray=True)
    size = 10
    def laplace(tau):
        gauss = G(size, tau)
        x_kernel = np.diff(np.diff(gauss, axis=0), axis=0)
        i_xx = convolve2d(image, x_kernel, mode="same")

        y_kernel = np.diff(np.diff(gauss, axis=1), axis=1)
        i_yy = convolve2d(image, y_kernel, mode="same")
        return tau * (i_xx + i_yy)

    n = 20
    taus = np.linspace(1, 100, n)
    ps = []
    for tau in taus:
        print("Doing tau", tau)
        i = laplace(tau)

        peaks = peak_local_max(i, num_peaks=150)
        maximas = np.array(list(zip(peaks, i[peaks[...,0], peaks[...,1]], itertools.repeat(tau))))
        ps.extend(maximas)

    plt.imshow(image, cmap="gray")
    top150 = np.array(sorted(ps, key=lambda p: abs(p[1]), reverse=True)[:150])
    for peaks, value, tau in top150:
        if not peaks.any():
            print("No peaks at tau", tau)
            continue

    plt.scatter([x for _, x in top150[:,0]], [y for y, _ in top150[:,0]], color='r', marker='o', s=list(top150[:,2]), alpha=0.3)
    plt.show()

if __name__ == '__main__':
    Exercise_2_4()