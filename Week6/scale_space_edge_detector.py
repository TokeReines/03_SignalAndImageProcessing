import numpy as np
import matplotlib.pyplot as plt

def task3_1(sigma=0.1):
    def _soft_edge(y, x, sigma=2):
        x = (x - ((len(x) - 1) / 2))
        left = 1 / (np.sqrt(2 * np.pi * sigma ** 2))
        right = np.exp(-((x ** 2) / (2 * sigma ** 2)))
        return left * right


    points = np.fromfunction(_soft_edge, (16, 16), dtype=float)
    img = np.cumsum(points, axis=1)
    plt.imshow(img, cmap='gray')

if __name__ == "__main__":
    task3_1()
    plt.show()