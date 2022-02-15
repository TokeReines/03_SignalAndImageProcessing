from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.pyplot import imshow




def gamma_transform(img, mode, *, gamma=0.45):
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


def task1_1():
    woman = Image.open("Week1/Images//woman.jpg").convert('L')

    fig, axs = plt.subplots(1, 5, figsize=(15, 8))
    axs[0].imshow(gamma_transform(woman, 'L', gamma=2).convert(mode='RGB'), cmap="gray", aspect='auto', vmax=255, vmin=0)
    axs[1].imshow(woman, cmap="gray", aspect='auto', vmax=255, vmin=0)
    axs[2].imshow(gamma_transform(woman, 'L', gamma=0.75), cmap="gray", aspect='auto', vmax=255, vmin=0)
    axs[3].imshow(gamma_transform(woman, 'L', gamma=0.5), cmap="gray", aspect='auto', vmax=255, vmin=0)
    axs[4].imshow(gamma_transform(woman, 'L', gamma=0.25), cmap="gray", aspect='auto', vmax=255, vmin=0)

    plt.show()


def task1_2():
    autumn = Image.open("Week1/Images//autumn.tif")
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    axs[0].imshow(autumn, aspect='auto')
    axs[1].imshow(gamma_transform(autumn, mode='RGB'), aspect='auto', interpolation='nearest')
    # gamma_transform(autumn).save("lol.jpg")

    plt.show()

def task1_3():
    autumn = Image.open("Week1/Images//autumn.tif")
    hsv_autumn = autumn.convert('HSV')
    gamma_corrected_hsv = gamma_transform(hsv_autumn, mode='HSV', gamma=0.45)
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    axs[0].imshow(autumn, aspect='auto')
    axs[1].imshow(gamma_corrected_hsv, aspect='auto', vmin=0, vmax=100)

    plt.show()

if __name__ == '__main__':
    task1_1()
    task1_2()
    task1_3()