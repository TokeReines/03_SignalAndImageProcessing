from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.pyplot import imshow




def gamma_transform(img, *, mode, gamma=0.45):
    """
    Performs gamma-correction
    Material: Computer Vision Szeliski ch. 2
    """
    if not isinstance(img, np.ndarray):
        try:
            img = np.array(img)
        except:
            print("Unknown input format <img>. Expected type ndarray or PIL image")

    dim = len(img.shape)
    mode = mode.upper()

    if dim == 3:
        if mode == 'RGB':
            # 3D array, slice notation below signifies slices of R, G, B for each pixel
            img[:, :, 0] = ((img[:, :, 0] / 255) ** gamma) * 255
            img[:, :, 1] = ((img[:, :, 1] / 255) ** gamma) * 255
            img[:, :, 2] = ((img[:, :, 2] / 255) ** gamma) * 255
        elif mode == 'HSV':
            # HSV gamma-correction for V-slice
            img[:, :, 2] = ((img[:, :, 1] / 100) ** gamma) * 100
        else:
            raise Exception("Bad image format")

    if dim == 2:
        # Grayscale case
        img = ((img / 255) ** gamma) * 255

    img[img < 0] = 0
    img[img > 255] = 255
    return Image.fromarray(img.astype('uint8'), mode=mode)


def cummulative_histogram(histogram):
    cumulative_histogram = np.cumsum(histogram)
    return cumulative_histogram / np.sum(histogram)

def floating_point_img(img, cdf):
    img = np.array(img)
    fp_img = img * cdf[img]
    return fp_img


def task1_1():
    woman = Image.open("Week1/Images//woman.jpg").convert('L')
    gammas = [0.25, 0.5, 0.75, 1, 2]
    gamma_women = [gamma_transform(woman, mode='L', gamma=gamma) for gamma in gammas]

    fig, axs = plt.subplots(1, 5, figsize=(15, 8))
    for i, gamma_woman in enumerate(gamma_women):
        axs[i].imshow(gamma_woman, cmap="gray", aspect='auto', vmax=255, vmin=0)
        axs[i].set_title(f"Gamma={gammas[i]}")
    fig.suptitle('Task 1.1, grayscale')
    fig.show()


def task1_2():
    autumn = Image.open("Week1/Images/autumn.tif")
    gamma_transform_autumn = gamma_transform(autumn, mode='RGB', gamma=0.45)

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    axs[0].imshow(autumn, aspect='auto')
    axs[0].set_title("Original image")

    axs[1].imshow(gamma_transform_autumn, aspect='auto', interpolation='nearest')
    axs[1].set_title("Gamma=0.45, RGB")
    fig.suptitle('Task 1.2')
    fig.show()

def task1_3():
    autumn = Image.open("Week1/Images/autumn.tif")
    hsv_autumn = autumn.convert('HSV')
    gamma_corrected_hsv = gamma_transform(hsv_autumn, mode='HSV', gamma=0.45)

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    axs[0].imshow(autumn, aspect='auto')
    axs[0].set_title("Original image")
    axs[1].imshow(gamma_corrected_hsv, aspect='auto', vmin=0, vmax=100)
    axs[1].set_title("Gamma=0.45, HSV")
    fig.suptitle('Task 1.3')
    fig.show()


def task2_1():
    pout = Image.open("Week1/Images/pout.tif")
    gray_pout = pout.convert('L')
    bins = range(256)
    hist, _ = np.histogram(gray_pout, bins=bins)
    cumhist = cummulative_histogram(hist)

    fig, axs = plt.subplots(1, 3, figsize=(15, 8))
    axs[0].imshow(gray_pout, cmap="gray", aspect='auto', vmax=255, vmin=0)
    axs[0].set_title("Original image")

    axs[1].hist(np.array(gray_pout).flatten(), bins=bins)
    axs[1].set_title("Histogram")

    axs[2].plot(cumhist)
    axs[2].set_title("CDF")
    fig.suptitle('Task 2.1')
    fig.show()

def task2_2():
    pout = Image.open("Week1/Images/pout.tif")
    gray_pout = pout.convert('L')

    bins = range(256)
    hist, _ = np.histogram(gray_pout, bins=bins)
    cumhist = cummulative_histogram(hist)
    fp_img = floating_point_img(gray_pout, cumhist)

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    axs[0].imshow(gray_pout, cmap="gray", aspect='auto', vmax=255, vmin=0)
    axs[0].set_title("Original image")

    axs[1].imshow(fp_img, cmap="gray", aspect='auto', vmax=255, vmin=0)
    axs[1].set_title("Floating Point image")

if __name__ == '__main__':
    task1_1()
    task1_2()
    task1_3()
    task2_1()
    task2_2()
    plt.show()