import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
from skimage import io

def task_2a():
    woman = io.imread("Week1/Images/trui.png", as_gray=True).astype(float)

    woman_fourier = np.fft.fftshift(np.fft.fft2(woman))

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].imshow(woman, cmap="gray", aspect='auto')
    axs[0].set_title("Original image")

    axs[1].imshow(100*np.log(1 + abs(woman_fourier)), cmap="gray", aspect="auto")
    axs[1].set_title("Power Spectrum")

    # axs[2].imshow(woman_power, cmap="gray", aspect="auto")
    # axs[2].set_title("FFTshift")
    fig.tight_layout()
    plt.show()

def task_3c():
    cameraman = io.imread("Week1/Images/cameraman.tif", as_gray=True)

    def _acos(img, v,w):
        img = np.copy(img)
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                img[x,y] *=  np.cos(v*x+ w*y)
        return img

    def _ftimg(img):
        return 100*np.log(1 + abs(img))


    v, w = 100, 14
    tmp = np.copy(np.array(cameraman))
    cam_fourier = np.fft.fftshift(np.fft.fft2(cameraman))
    acos_tmp = _acos(tmp, v, w)
    acos_fourier = np.fft.fftshift(np.fft.fft2(acos_tmp))

    rows, cols = np.size(cam_fourier, 0), np.size(cam_fourier, 1)
    crow, ccol = rows//2, cols//2

    def _filter(f):
        rad = 50
        fil = np.zeros_like(f)
        tmp = np.copy(f)
        n, m = f.shape

        power_spectrum_f = np.fft.fftshift(np.fft.fft2(tmp))
        mag = v*np.log(abs(power_spectrum_f))

        for i in range(n):
            for j in range(m):
                fil[i,j] = f[i,j] / np.cos(v*i + w*j)

        fil[n-1:n,m-1:m+1] = 1
        f_fil = np.fft.fftshift(np.fft.fft2(fil))
        power_spectrum_f_filtered = np.multiply(power_spectrum_f, fil)
        power_spectrum_f_filtered = np.fft.ifftshift(power_spectrum_f_filtered)
        f_filtered = np.fft.ifft2(power_spectrum_f_filtered)

        return np.real(f_filtered) 
    img_back = _filter(acos_tmp)
    img_f = np.fft.fftshift(np.fft.fft2(img_back))
    print(img_back)

    fig, axs = plt.subplots(2, 3, figsize=(12, 12))

    axs[0,0].imshow(cameraman, cmap="gray", aspect='auto')
    axs[0,0].set_title("Original image")

    axs[0,1].imshow(acos_tmp, cmap="gray", aspect="auto")
    axs[0,1].set_title("Planarwaves")

    axs[0,2].imshow(-abs(img_back), cmap="gray", aspect="auto")
    axs[0,2].set_title("Restored")

    axs[1,0].imshow(_ftimg(cam_fourier), cmap="gray", aspect='auto')
    axs[1,0].set_title("Original Power Spectrum")

    axs[1,1].imshow(_ftimg(acos_fourier), cmap="gray", aspect="auto")
    axs[1,1].set_title("Planarwaves Power Spectrum")
    
    axs[1,2].imshow(_ftimg(img_f), cmap="gray", aspect="auto")
    axs[1,2].set_title("Restored Power Spectrum")

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    task_3c()