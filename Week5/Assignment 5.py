# Assignment


# Packages

# Numpy, Matplotlib and miscellaneous
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
from scipy.signal import convolve2d, gaussian, fftconvolve
from scipy.ndimage import median_filter, gaussian_filter, convolve
from scipy import interpolate
from scipy import ndimage
from scipy.ndimage import shift


# Transformations on images - Translation

# 4.

def white_square(d1, d2):
    """ Inputs: d1 and d2 has to be odd dimensions.
        Output: Black image with a white dot in the center. """
    I = np.zeros((d1, d2))
    I[d1 // 2, d2 // 2] = 1
    return I


I = white_square(5, 5)
fig, axs = plt.subplots(1, figsize=(15, 6), facecolor='w', edgecolor='k')
imshow(I, cmap="gray")
show()


# 5.

def filter_translation(I, dx, dy):
    I = I.astype(np.float64)
    I_shift = fftconvolve(I, dx, mode="same")
    I_shift = fftconvolve(I_shift, dy, mode="same")
    return I_shift


I = white_square(5, 5)
dx_1 = np.array([[0, 0, 1]])
dx_2 = np.array([[1, 0, 0]])
dy_1 = np.array([1, 0, 0]).reshape(-1, 1)
dy_2 = np.array([0, 0, 1]).reshape(-1, 1)
I_1 = filter_translation(I, dx_1, dy_1)
I_2 = filter_translation(I, dx_2, dy_2)

fig, axs = plt.subplots(1, 3, figsize=(15, 6), facecolor='w', edgecolor='k')
axs = axs.ravel()
axs[0].imshow(I, cmap="gray")
axs[0].set_title("Original Image")
axs[1].imshow(I_1, cmap="gray")
axs[1].set_title("Translation one to right and one up")
axs[2].imshow(I_2, cmap="gray")
axs[2].set_title("Translation one to left and one down")
show()


# 6.
def space_translation(I, dx=0.6, dy=1.2):
    # https://theailearner.com/tag/cv2-warpaffine/
    # https://kwojcicki.github.io/blog/NEAREST-NEIGHBOUR

    rows, cols = I.shape
    # Transformation matrix
    M = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    orig_coord = np.indices((cols, rows)).reshape(2, -1)
    # Vertical stacking.
    orig_coord_f = np.vstack((orig_coord, np.ones(rows * cols)))
    print(orig_coord_f.shape)
    transform_coord = np.dot(M, orig_coord_f)
    # Rounding towards zero (Nearest Neighbour Interpolation)
    transform_coord = transform_coord.astype(np.int32)
    # Keep only the coordinates that fall within the image boundary.
    indices = np.all((transform_coord[1] < rows, transform_coord[0] < cols, transform_coord[1] >= 0, transform_coord[0] >= 0), axis=0)
    # Create a zeros image and project the points
    I_new = np.zeros_like(I)
    I_new[transform_coord[1][indices], transform_coord[0][indices]] = I[orig_coord[1][indices], orig_coord[0][indices]]
    return I_new


I = white_square(5, 5)
I_t = space_translation(I, 0.6, 1.2)

fig, axs = plt.subplots(1, 2, figsize=(15, 6), facecolor='w', edgecolor='k')
axs = axs.ravel()
axs[0].imshow(I, cmap="gray")
axs[0].set_title("Original Image")
axs[1].imshow(I_t, cmap="gray")
axs[1].set_title("Translating by $t =(0.6, 1.2)^{T}$")
show()


# 7.

def Fourier_translation(I, dx, dy):

    input_ = np.fft.fft2(I)
    freqs = np.fft.fftfreq(len(I))
    result = np.array([np.exp(-complex(0, 1) * 2 * np.pi * f * dx) for f in freqs]) * input_
    result = np.array([np.exp(-complex(0, 1) * 2 * np.pi * f * dy) for f in freqs]).reshape(-1, 1) * result
    result = np.fft.ifft2(result)
    return result.real


I = white_square(5, 5)
I_F = Fourier_translation(I, -1, 1)
I_shift = space_translation(I, -1, 1)

fig, axs = plt.subplots(1, 3, figsize=(15, 6), facecolor='w', edgecolor='k')
axs = axs.ravel()
axs[0].imshow(I, cmap="gray")
axs[0].set_title("Original Image")
axs[1].imshow(I_shift, cmap="gray")
axs[1].set_title("Translating two left and two down \n (Space domain)")
axs[2].imshow(I_F, cmap="gray")
axs[2].set_title("Translating two left and two down \n (Fourier method)")
show()


# 8.


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.fourier_shift.html

I = white_square(5, 5)
I_F = Fourier_translation(I, 0.6, 1.2)
I_shift = scipy.ndimage.fourier_shift(I, (0.6, 1.2))

fig, axs = plt.subplots(1, 3, figsize=(15, 6), facecolor='w', edgecolor='k')
axs = axs.ravel()
axs[0].imshow(I, cmap="gray")
axs[0].set_title("Original Image")
axs[1].imshow(I_F, cmap="gray")
axs[1].set_title("Translating by $t =(0.6, 1.2)^{T}$ (Fourier method)")
axs[2].imshow(I_F, cmap="gray")
axs[2].set_title("Translating by $t =(0.6, 1.2)^{T}$ (fourier_shift)")
show()


# Cameraman.
I_cam = imread("cameraman.tif")
I_F = Fourier_translation(I_cam, 0.6, 1.2)

fig, axs = plt.subplots(1, 2, figsize=(15, 6), facecolor='w', edgecolor='k')
axs[0].imshow(I_cam, cmap="gray")
axs[0].set_title("Cameraman")
axs[1].imshow(I_F, cmap="gray")
axs[1].set_title("Translating by $t =(0.6, 1.2)^{T}$ (Fourier method)")
show()

# Circles.
I_circles = imread("circles.png")
I_F = Fourier_translation(I_trui, 0.6, 1.2)

fig, axs = plt.subplots(1, 2, figsize=(15, 6), facecolor='w', edgecolor='k')
axs[0].imshow(I_trui, cmap="gray")
axs[0].set_title("Circles")
axs[1].imshow(I_F, cmap="gray")
axs[1].set_title("Translating by $t =(0.6, 1.2)^{T}$ (Fourier method)")
show()
