import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def G(size, sigma):
    #size = abs(int(2 * sigma))
    # https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    ax = np.linspace(-(size - 1) // 2, (size - 1) // 2, size)
    gauss = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def task3_1():
    def _soft_edge(size, sigma=2):
        ax = np.meshgrid(range(size), range(size), indexing='xy')
        x = [(x - ((size - 1) / 2)) for x in ax]
        gauss = [(1 / (np.sqrt(2 * np.pi * sigma ** 2))) * np.exp(-((x ** 2) / (2 * sigma ** 2))) for x in x[0]]
        return np.cumsum(gauss, axis=1)


    size = 32
    taus = [1, 1,2,5,10,15,20,30]
    img = _soft_edge(size)

    

    fig, axs = plt.subplots(2, 4, figsize=(15, 6))
    

    for i, tau in enumerate(taus):
        img_scale_space = convolve2d(img, G(size, tau), mode="same", boundary='symm')
        if i > 3:
            axs[1][i-8].imshow(img_scale_space, cmap='gray')
            axs[1][i-8].set_title("J(x,y,{})".format(tau))
            axs[1][i-8].axis("off")
        else:
            axs[0][i].imshow(img_scale_space, cmap='gray')
            axs[0][i].set_title("J(x,y,{})".format(tau))
            axs[0][i].axis("off")

            
    a = axs[0][0].imshow(img, cmap='gray')
    fig.colorbar(a, ax=axs[0][0])
    axs[0][0].set_title("S(x,y)")


if __name__ == "__main__":
    task3_1()
    plt.show()