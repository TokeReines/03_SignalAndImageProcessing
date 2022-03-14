import skimage
import matplotlib.pyplot as plt
import torch 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import procrustes


def setup():
    X_train = np.loadtxt('diatom/SIPdiatomsTrain.txt', delimiter = ',')
    X_test = np.loadtxt('diatom/SIPdiatomsTest.txt', delimiter = ',')
    X_train_classes = np.loadtxt('diatom/SIPdiatomsTrain_classes.txt', delimiter = ',')
    X_test_classes = np.loadtxt('diatom/SIPdiatomsTest_classes.txt', delimiter = ',')
    return X_train, X_test, X_train_classes, X_test_classes

def task2_1():
    X_train, X_test, X_train_y, X_test_y = setup()

    target_x =  X_train[0, ::2] 
    target_y = X_train[0, 1::2]
    target_points = np.stack((target_x, target_y), axis=1)

    train_xs = X_train[1:, ::2]
    train_ys = X_train[1:, 1::2]
    train_points = np.stack((train_xs, train_ys), axis=2)

    mtx1_0, mtx2_0, disparity_0 = procrustes(target_points, train_points[0])
    mtx1_1, mtx2_1, disparity_1 = procrustes(target_points, train_points[1])
    mtx1_2, mtx2_2, disparity_2 = procrustes(target_points, train_points[2])
    target_standard, _, _ = procrustes(target_points, target_points)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].scatter(target_x, target_y, color='r', marker='D')
    axs[1].scatter(target_x, target_y, color='r', marker='D', alpha=.5)
    axs[2].scatter(target_standard[:,0], target_standard[:,1], color='r', marker='D', alpha=.25)
    axs[0].set_title("Target diatom")

    axs[1].scatter(train_xs[0], train_ys[0], color='b', marker='1')
    axs[1].scatter(train_xs[1], train_ys[1], color='g', marker='2')
    axs[1].scatter(train_xs[2], train_ys[2], color='y', marker='3')
    axs[1].set_title("Procrustes Diatoms Original")

    axs[2].scatter(mtx2_0[:,0], mtx2_0[:,1], color='y', marker='3')
    axs[2].scatter(mtx2_1[:,0], mtx2_1[:,1], color='b', marker='1')
    axs[2].scatter(mtx2_2[:,0], mtx2_2[:,1], color='g', marker='2')
    axs[2].set_title("Post Procrustes Diatoms")
    fig.suptitle("Procrustes Transformation with target diatom in red")
    fig.tight_layout()

if __name__ == '__main__':
    task2_1()

    plt.show()
