# Assignment 3

# Packages

# Numpy, Matplotlib and miscellaneous
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis, imshow, show, title, colorbar, figure
import timeit
from timeit import default_timer as timer
import time
from numpy import linalg

# Scikit-image.
from skimage import img_as_ubyte, img_as_float
from skimage.io import imread
from skimage.color import rgb2gray, hsv2rgb, rgb2hsv
from skimage.util import random_noise
from skimage import exposure
from skimage.filters import gaussian, rank


# Scipy.

from scipy import stats
from scipy.signal import convolve2d
from scipy.ndimage import median_filter, gaussian_filter
from scipy import interpolate
import scipy.fft


# Theory


# Exercise (d) iv..


def Exercise_d():

    def b_a(x, a=1):
        if np.abs(x) <= a / 2:
            return 1 / a
        else:
            return 0

    def B_a(k, a=1):
        return (1 / a * k * np.pi) * np.sin(a * k * np.pi)

    a = [1, 2, 3, 4]
    # x = np.linspace(-5, 5, 100)
    k = np.linspace(-2, 2, 200)
    #L_space = [b_a(i,a) for i in x]
    L_1 = [B_a(j, a[0]) for j in k]
    L_2 = [B_a(j, a[1]) for j in k]
    L_3 = [B_a(j, a[2]) for j in k]
    L_4 = [B_a(j, a[3]) for j in k]
    plt.plot(k, L_1, color='red', label="$a=1$")
    plt.plot(k, L_2, color="orange", label="$a=2$")
    plt.plot(k, L_3, color='blue', label="$a=3$")
    plt.plot(k, L_4, color="green", label="$a=4$")
    plt.xlim([-1, 1])
    plt.ylim([-2.2, 2.2])
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.xlabel("k")
    plt.ylabel("$B_{a}(k)$")
    # plt.title("$B_{a}(k)$ for different values of $a$")
    plt.tight_layout()
    plt.show()


Exercise_d()

# Practice

## Exercise (a)


def ps(I, t):
    f = scipy.fft.fft2(I)
    fshift = scipy.fft.fftshift(f)
    if t == 1:
        return np.log(np.square(np.abs(fshift)))
    else:
        return np.square(np.abs(fshift))


def Exercise_a():
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft2.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fftshift.html
    # https://docs.opencv.org/3.4/de/dbc/tutorial_py_fourier_transform.html

    # Image and transformation
    I = imread('trui.png')

    power_spectrum = ps(I, 0)
    log_power_spectrum = ps(I, 1)

    # Plotting

    fig, axs = plt.subplots(1, 3, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.001)
    # fig.suptitle("Power spectrum representation of image", fontsize=16)
    axs = axs.ravel()

    axs[0].imshow(I, cmap='gray')
    axs[0].set_title("Input image")
    axs[0].axis("off")

    axs[1].imshow(power_spectrum, cmap='gray')
    axs[1].set_title("Power spectrum")
    axs[1].axis("off")

    axs[2].imshow(log_power_spectrum, cmap='gray')
    axs[2].set_title("log transformed power spectrum")
    axs[2].axis("off")

    plt.show()


Exercise_a()

## Exercise (b)

def convo(I, mask):

    # DFT and shift of frequencies.
    f = scipy.fft.fft2(I)
    fshift = scipy.fft.fftshift(f)

    # Apply mask and inverse DFT
    product = fshift * mask
    f_ishift = np.fft.ifftshift(product)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back


## Exercise (c)

# Cosine wave.
def wave(I, a_0=1, v_0=1, w_0=1):
    x = np.arange(0, I.shape[0], 1)
    y = np.arange(0, I.shape[1], 1)
    X, Y = np.meshgrid(x, y)
    W = a_0 * np.cos(v_0 * X + w_0 * Y)
    return I + W


I = img_as_float(imread("cameraman.tif"))
I_1 = wave(I, 1, 1, 0)
I_2 = wave(I, 1, 0, 1)
I_3 = wave(I, 1, 1, 1)
I_cos = [I_1, I_2, I_3]
PS_1 = ps(I_1, 0)
PS_2 = ps(I_2, 0)
PS_3 = ps(I_3, 0)
PS = [PS_1, PS_2, PS_3]
log_PS_1 = ps(I_1, 1)
log_PS_2 = ps(I_2, 1)
log_PS_3 = ps(I_3, 1)
log_PS = [log_PS_1, log_PS_2, log_PS_3]

# Plotting

fig, axs = plt.subplots(3, 3, figsize=(15, 15), facecolor='w', edgecolor='k')
# fig.suptitle('Power spectrum of image \n with added cosine wave given by $a_{0}\cos(v_{0}x + w_{0}y)$', fontsize=16)
L = [1, 2, 3, 4, 5, 6, 7, 8, 9]

for k in L:
    plt.subplot(3, 3, k)

    # Image with added cosine wave.
    if k == 1:
        plt.subplot(3, 3, k)
        plt.imshow(I_cos[0], cmap="gray")
        plt.title("$ I + \cos(x)$")
    if k == 4:
        plt.subplot(3, 3, k)
        plt.imshow(I_cos[1], cmap="gray")
        plt.title("$ I + \cos(y)$")
    if k == 7:
        plt.subplot(3, 3, k)
        plt.imshow(I_cos[2], cmap="gray")
        plt.title("$I + \cos(x+y)$")

    # Power spectrum.
    if k == 2:
        plt.imshow(PS[0], cmap='gray')
        plt.title("$| F(u) |^2$")
    if k == 5:
        plt.imshow(PS[1], cmap='gray')
        plt.title("$| F(u) |^2$")
    if k == 8:
        plt.imshow(PS[2], cmap='gray')
        plt.title("$| F(u) |^2$")

    # Log transformation of power spectrum.
    if k == 3:
        plt.imshow(log_PS[0], cmap='gray')
        plt.title("$\log(| F(u) |^2$")
    if k == 6:
        plt.imshow(log_PS[1], cmap='gray')
        plt.title("$\log(| F(u) |^2$")
    if k == 9:
        plt.imshow(log_PS[2], cmap='gray')
        plt.title("$\log(| F(u) |^2$")

    plt.axis("off")

plt.show()


def LPF(I, a, b):
    # https://docs.opencv.org/3.4/de/dbc/tutorial_py_fourier_transform.html
    f = scipy.fft.fft2(I)
    fshift = scipy.fft.fftshift(f)
    rows, cols = I.shape
    crow, ccol = rows // 2, cols // 2
    # Create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - a:crow + a, ccol - b:ccol + b] = 1

    # Apply mask and inverse DFT
    product = fshift * mask
    f_ishift = np.fft.ifftshift(product)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back


I = img_as_float(imread("cameraman.tif"))
I_1 = I + wave(I, 1, 1, 0)
I_2 = I + wave(I, 1, 0, 1)
I_3 = I + wave(I, 1, 1, 1)
I_cos = [I_1, I_2, I_3]

# Plotting

fig, axs = plt.subplots(3, 3, figsize=(10, 15), facecolor='w', edgecolor='k')
# fig.subplots_adjust(hspace = .5, wspace=.001)
# fig.suptitle('Power spectrum of image \n with added cosine wave given by $a_{0}\cos(v_{0}x + w_{0}y)$', fontsize=16)
L = [1, 2, 3, 4, 5, 6]

for k in L:
    plt.subplot(3, 2, k)

    if k == 1:
        plt.imshow(I_cos[0], cmap="gray")
        plt.title("$I + \cos(x)$")
    if k == 3:
        plt.imshow(I_cos[1], cmap="gray")
        plt.title("$I + \cos(y)$")
    if k == 5:
        plt.imshow(I_cos[2], cmap="gray")
        plt.title("$I + \cos(x+y)$")
    if k == 2:
        plt.imshow(LPF(I_cos[0], 30, 30), cmap='gray')
        plt.title("Filtered image of $I + \cos(x)$")
    if k == 4:
        plt.imshow(LPF(I_cos[1], 30, 30), cmap='gray')
        plt.title("Filtered image of $I + \cos(y)$")
    if k == 6:
        plt.imshow(LPF(I_cos[2], 30, 30), cmap='gray')
        plt.title("Filtered image of $I + \cos(x+y)$")

    plt.axis("off")

plt.show()
