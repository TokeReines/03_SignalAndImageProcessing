# Assignment 4 Individual
from skimage.util import random_noise
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.io import imread
from skimage import img_as_ubyte, img_as_float
from scipy.signal import convolve2d
from skimage.morphology import binary_dilation

# From Assignment 1
# def random_noise(img, mode='gauss', noise_percentage=0.08, var=0.01):
#     mode = str(mode).upper()

#     if not isinstance(img, np.ndarray):
#         try:
#             img = np.array(img).astype('float64')
#         except:
#             print("Unknown input format <img>. Expected type ndarray or PIL image")
   
#     if img.min() < 0:
#         low_clip = -1.
#     else:
#         low_clip = 0.

#     row, col = img.shape
#     pixels = row * col
#     if mode == 'SP':
#         amount_random_pixels = round(pixels * noise_percentage)
#         for i in range(amount_random_pixels):
#             random_pixel = (random.randint(0, row - 1), random.randint(0, col - 1))
#             img[random_pixel] = random.choice((0, 255))

#     elif mode == 'GAUSS':
#         rng = np.random.default_rng(42)
#         mean = 0
#         noise =  rng.normal(mean, var**0.5, (row, col)) 
#         out = img + noise
       
#     img = np.clip(out, low_clip, 1.0) 
#     return img 

def task1_3():
    def _sobel_filter(img, mode='both'):
        filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        filter_y = np.flip(filter_x.T, axis=0)
        if mode == 'horizontal':
            img_sobel = convolve2d(img, filter_x)
        if mode == 'vertical':
            img_sobel = convolve2d(img, filter_y)
        else:
            x = convolve2d(img, filter_x)
            y = convolve2d(img, filter_y)
             # Hypot = sqrt(x1**2 + x2**2)
            gradient_mag = np.hypot(x, y)
            #gradient_mag *=  (255. / gradient_mag.max()) ** 2
            img_sobel = gradient_mag

        return img_sobel

    def _prewitt_filter(img, mode='both'):
        filter_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        filter_y = np.flip(filter_x.T, axis=0)
        if mode == 'horizontal':
            img_prewitt = convolve2d(img, filter_x)
        if mode == 'vertical':
            img_prewitt = convolve2d(img, filter_y)
        else:
            x = convolve2d(img, filter_x)
            y = convolve2d(img, filter_y)
            # Hypot = sqrt(x1**2 + x2**2)
            gradient_mag = np.hypot(x, y)
            #gradient_mag = np.sqrt(np.square(x) + np.square(y))
            #radient_mag *=  (255. / gradient_mag.max()) ** 2
            img_prewitt = gradient_mag
        return img_prewitt


    eight = imread("Week1/Images/eight.tif")
    vars = [.001, 0.01, 0.1, 0.5]
    fig, axs = plt.subplots(4, 3, figsize=(8, 16))
    for i, var in enumerate(vars):
        eight_noisy_gauss = random_noise(eight, mode='gaussian', var=var)

        _sobel_gradient_mag = _sobel_filter(eight_noisy_gauss)
        _prewitt_gradient_mag = _prewitt_filter(eight_noisy_gauss)

        axs[i][0].imshow(eight_noisy_gauss, cmap="gray", aspect='auto')
        axs[i][0].set_title("Gauss Noise with variance={}".format(var)) 

        axs[i][1].imshow(_sobel_gradient_mag, cmap="gray", aspect='auto')
        axs[i][1].set_title("Sobel Gradient Magnitude") 

        axs[i][2].imshow(_prewitt_gradient_mag, cmap="gray", aspect='auto')
        axs[i][2].set_title("Prewitt Gradient Magnitude") 

    fig.tight_layout()
    retval = _sobel_filter(eight_noisy_gauss)

    return retval

def task4_1():
    digit = imread("Week1/Images/8.bmp")
    digit[digit>0] = 1

   
    mask_t = np.array([[1,1,1],[0,1,0],[0,1,0]]) 
    mask_tile = np.array([[0,1],[1,0]])
    mask_worm = np.array([0,0,1])

    eight_t_dilated = binary_dilation(digit, selem=mask_t)
    eight_tile_dilated = binary_dilation(digit, selem=mask_tile)
    eight_flat = digit.reshape(1, digit.size)
    eight_dilated = binary_dilation(eight_flat, selem=mask_worm.reshape(1,3)).reshape(7,7)
    
    fig, axs = plt.subplots(1, 3, figsize=(7, 7))
    axs[0].imshow(eight_t_dilated, cmap="gray", aspect='auto')
    axs[0].set_title("Gauss Noise with variance")
    axs[1].imshow(eight_tile_dilated, cmap="gray", aspect='auto')
    axs[1].set_title("Sobel Gradient Magnitude")
    axs[2].imshow(eight_dilated, cmap="gray", aspect='auto')
    axs[2].set_title("Prewitt Gradient Magnitude") 
    fig.tight_layout()

  


if __name__=="__main__":
    #task1_3()
    task4_1()
    plt.show()