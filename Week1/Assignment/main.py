import math
import timeit

from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.pyplot import imshow
from pyasn1.type.univ import Integer


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


def cummulative_histogram(histogram) -> np.array:
    cumulative_histogram = np.cumsum(histogram)
    return cumulative_histogram / int(cumulative_histogram.max())  # Equal to x / NM (dimensions of image) for each x in the histogram


def histogram_match(image1, image2):
    bins = range(256)

    hist1, _ = np.histogram(image1, bins=bins)
    cdf1 = cummulative_histogram(hist1)

    hist2, _ = np.histogram(image2, bins=bins)
    cdf2 = cummulative_histogram(hist2)
    for i, x in enumerate(cdf1):
        hist2[i] = pesudo_inverse_floating_point_img(cdf2, x)

    corrected_cdf = cummulative_histogram(hist2)
    c2 = floating_point_img(image2, corrected_cdf)
    return c2
    # interp = np.interp(c1, c2_inverse, np.unique(image2, return_counts=True)[0])
    # im2 = interp(im.flatten(), bins[:-1], cdf)

def floating_point_img(img, cdf) -> np.array:
    img = np.array(img)
    fp_img = cdf[img]
    return fp_img

def pesudo_inverse_floating_point_img(cumhist, probability: float) -> int:
    """
    The CDF is in general not invertible due to it's possible plateaus, which would result in a range of intensities.
    To accommodate this, we extract the minimum intensity of the possible plateau.
    Arguments:
        cumhist: CDF, an array of length 0-255 where each x is from [0, 1].
        probability
    Return:
        Intensity from a given probability (probability) inversely looked up in cumhist
    """
    a = np.min(np.where(cumhist >= probability))
    return int(a)


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
    pout = Image.open("Week1/Images/pout.tif").convert('L')
    bins = range(256)
    hist, _ = np.histogram(pout, bins=bins)
    cumhist = cummulative_histogram(hist)

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    axs[0].imshow(pout, cmap="gray", aspect='auto', vmax=255, vmin=0)
    axs[0].set_title("Original image")

    axs[1].plot(cumhist, color='b')
    axs[1].hist(np.array(pout).flatten(), bins=bins, density=True)
    axs[1].set_title("Histogram and cdf")
    axs[1].set_xlabel('Intensity value')
    axs[1].set_ylabel('Fraction of pixels')

    fig.suptitle('Task 2.1')
    fig.show()

def task2_2():
    pout = Image.open("Week1/Images/pout.tif")
    gray_pout = pout.convert('L')

    bins = range(256)
    hist, _ = np.histogram(gray_pout, bins=bins)
    cumhist = cummulative_histogram(hist)
    fp_img = floating_point_img(gray_pout, cumhist)

    fig, axs = plt.subplots(1, 4, figsize=(15, 8))
    axs[0].imshow(gray_pout, cmap="gray", aspect='auto', vmax=255, vmin=0)
    axs[0].set_title("Original image")

    axs[1].hist(np.array(gray_pout).flatten(), bins=bins)
    axs[1].set_title("Original Histogram")

    axs[2].imshow(fp_img * 255, cmap="gray", aspect='auto', vmax=255, vmin=0)
    axs[2].set_title("Floating Point image")

    axs[3].hist(np.array(fp_img).flatten() * 255, bins=bins)
    axs[3].set_title("Floating Point Histogram")
    fig.show()

def task2_3():
    pout = Image.open("Week1/Images/pout.tif")
    gray_pout = pout.convert('L')

    bins = range(256)
    hist, _ = np.histogram(gray_pout, bins=bins)
    cumhist = cummulative_histogram(hist)
    fp_img = pesudo_inverse_floating_point_img(cumhist, 0.6)


def task2_4():
    src = Image.open("Week1/Images/autumn.tif").convert('L')
    templ = Image.open("Week1/Images/pout.tif").convert('L')

    bins = range(256)

    hist_src, _ = np.histogram(src, bins=bins)
    hist_templ, _ = np.histogram(templ, bins=bins)

    cdf_src = cummulative_histogram(hist_src)
    cdf_templ = cummulative_histogram(hist_templ)

    cdf_templ_inverse = hist_templ.copy()
    for i, x in enumerate(cdf_src):
        cdf_templ_inverse[i] = pesudo_inverse_floating_point_img(cdf_templ, x)

    p = 0.50
    pp = math.floor(p*255)
    color = cdf_templ_inverse[pp]

    c2 = floating_point_img(src, cdf_src)
    hist2_inverse, _ = np.histogram(c2 * 255, bins=bins)
    corrected_cdf = cummulative_histogram(hist2_inverse)
    timeit.timeit()
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs[0][0].imshow(src, cmap="gray", aspect='auto', vmax=255, vmin=0)
    axs[0][0].set_title("Source")

    axs[0][1].imshow(templ, cmap="gray", aspect='auto', vmax=255, vmin=0)
    axs[0][1].set_title("Template")

    axs[0][2].imshow(c2 * 255, cmap="gray", aspect='auto', vmax=255, vmin=0)
    axs[0][2].set_title("Matched")

    l1, = axs[1][0].plot(cdf_src, color='b', label="Source")
    l2,  = axs[1][0].plot(cdf_templ, color='r', label="Template")
    # l3, = axs[1][0].plot(cdf_src_inverse * 255, color='g', label="Inverse")
    axs[1][0].set_title("CDF")
    axs[1][0].set_ylabel("Cumulative intensity")
    axs[1][0].set_xlabel("Pixel intensitiy")
    axs[1][0].legend(handles=[l1, l2])

    l1, = axs[1][1].plot(cdf_src, color='b', label="Source")
    l2,  = axs[1][1].plot(cdf_templ, color='r', label="Template")
    # l3, = axs[1][0].plot(cdf_src_inverse * 255, color='g', label="Inverse")
    axs[1][0].set_title("CDF")
    axs[1][0].set_ylabel("Cumulative intensity")
    axs[1][0].set_xlabel("Pixel intensitiy")
    axs[1][0].legend(handles=[l1, l2])

    #l1, = axs[1][1].plot(hist_src, color='b', label="Source")
    # l2,  = axs[1][1].plot(hist_templ, color='r', label="Template")
    # l3, = axs[1][1].plot(hist_templ_inverse, color='g', label="Inverse")
    # axs[1][1].set_title("Histograms")
    # axs[1][1].set_ylabel("Cumulative %")
    # axs[1][1].set_xlabel("Pixel intensitiy")
    # axs[1][1].legend(handles=[l2, l3])

    fig.show()



if __name__ == '__main__':
    # task1_1()
    # task1_2()
    # task1_3()
    #
    # task2_1()
    # task2_2()
    # task2_3()
    task2_4()
    plt.show()