import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.signal import convolve2d
from skimage.feature import peak_local_max
from skimage.io import imread


def G(size, sigma):
    #size = abs(int(2 * sigma))
    # https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    ax = np.linspace(-(size - 1) // 2, (size - 1) // 2, size)
    gauss = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def soft_edge(size, sigma=2):
    ax = np.meshgrid(range(size), range(size), indexing='xy')
    x = [(x - ((size - 1) / 2)) for x in ax]
    gauss = [(1 / (np.sqrt(2 * np.pi * sigma ** 2))) * np.exp(-((x ** 2) / (2 * sigma ** 2))) for x in x[0]]
    return np.cumsum(gauss, axis=1)

def task3_1():
    size = 32
    taus = [1, 1,2,5,10,15,20,30]
    img = soft_edge(size)

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

def task3_3():
    image = imread("Week6/images/hand.tiff", as_gray=True)

    size = 20
    gamma = 1/2

    def snsgm_operator(image, size, tau):
        gauss = G(size, tau)
        x_kernel = np.diff(gauss, axis=0)
        i_xx = convolve2d(image, x_kernel, mode="same", boundary='symm')

        y_kernel = np.diff(gauss, axis=1)
        i_yy = convolve2d(image, y_kernel, mode="same", boundary='symm')
        tau_sq_gamma = tau ** (2*gamma)
        return (tau_sq_gamma * i_xx**2) + (tau_sq_gamma * i_yy**2)

    fig, ax = plt.subplots(1, figsize=(15, 6), facecolor='w', edgecolor='k')

    num_peaks = 200
    steps = 20
    maxs = []
    taus = np.logspace(-2, 3, steps, base=2)
    for tau in taus:
        #fig2, ax2 = plt.subplots(1)
        #ax2.set_title(tau)

        edge_img = snsgm_operator(image, size, tau)
        max_peaks = peak_local_max(edge_img, num_peaks=num_peaks/steps)
        maxs.extend(max_peaks)
        for peak in max_peaks:
            ax.add_patch(Circle((peak[1], peak[0]), tau, fill=False, color='red'))
            #ax2.add_patch(Circle((peak[1], peak[0]), tau, fill=False, color='red'))

        #ax2.scatter(max_peaks[:,1], max_peaks[:,0], color='r', marker='o')
        #imshow = ax2.imshow(edge_img, cmap="gray")
        #fig.colorbar(imshow, ax=ax2)

    maxs = np.array(maxs)
    imshow = ax.imshow(image, cmap="gray")
    s1 = ax.scatter(maxs[:,1], maxs[:,0], color='r', marker='o', label="Edges")
    ax.legend(handles=[s1])
    fig.colorbar(imshow, ax=ax)

if __name__ == "__main__":
    task3_3()
    plt.show()