from time import time

from PIL import Image
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift
import numpy as np
import matplotlib.pyplot as plt
import skimage
import matplotlib
from scipy.signal import fftconvolve
from skimage import io
from skimage import color
from matplotlib import cm

matplotlib.rcParams['figure.figsize'] = (16, 16)


def convolve2d(image, kernel):
    h, w = kernel.shape
    h = h // 2
    w = w // 2
    convolved_img = np.zeros(image.shape)

    for x in range(h, image.shape[0] - h):
        for y in range(w, image.shape[1] - w):
            sum = 0
            for m in range(kernel.shape[0]):
                for n in range(kernel.shape[1]):
                    sum += kernel[m][n] * image[x - h + m][y - w + n]
            convolved_img[x][y] = sum

    return convolved_img


def convolvefft(image, kernel):
    def _pad_kernel():
        sz = (image.shape[0] - kernel.shape[0], image.shape[1] - kernel.shape[1])  # total amount of padding
        return ifftshift(np.pad(kernel, (((sz[0] + 1) // 2, sz[0] // 2), ((sz[1] + 1) // 2, sz[1] // 2)),
                                'constant'))

    padded_kernel = _pad_kernel()
    image_fft = fft2(image)
    image = ifft2(image_fft * mask)




    kernel_fft = fft2(padded_kernel)
    conv_fft = image_fft * kernel_fft
    conv = np.absolute(np.real(ifft2(conv_fft)))
    return conv

def mask(image, x, y, v, w):
    mymask = image.copy()
    for i, _ in enumerate(mymask):
        for j, _ in enumerate(x):
            if i >= x and i <=
    np.where()


if __name__ == '__main__':
    image = np.array(Image.open("Week1/Images/trui.png").convert('L'))
    ks = [3, 7, 13, 21]
    n = 1

    fig, axs = plt.subplots(len(ks), 4, figsize=(24, 13))
    for i, k in enumerate(ks):
        kernel = np.ones([k, k], dtype=int) / (k * k)
        time1 = 0
        time2 = 0
        for _ in range(n):
            t = time()
            c2d = convolve2d(image, kernel)
            time1 += time() - t

            t = time()

            fft = convolvefft(image, kernel)
            time2 += time() - t
        time1 /= n
        time2 /= n
        print(f"Convolving spatially with kernel size {k} took on average:", time1)
        print(f"Convolving in Fourier with kernel size {k} took on average:", time2)

        axs[0, i].imshow(image, cmap="gray", aspect='auto', vmax=255, vmin=0)
        axs[0, i].set_title(f"Original image, kernel size {k}")
        axs[1, i].imshow(c2d, cmap="gray", aspect='auto', vmax=255, vmin=0)
        axs[1, i].set_title("Convolved 2 For-loops,")
        axs[2, i].imshow(fft, cmap="gray", aspect='auto', vmax=255, vmin=0)
        axs[2, i].set_title("Convolved Fourier")
        axs[3, i].imshow(fft - c2d, cmap="gray", aspect='auto', vmax=255, vmin=0)
        axs[3, i].set_title("Diff (Fourier - Spatial)")

    plt.show()
    a = 2
