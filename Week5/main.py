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

if __name__ == '__main__':
    X_train, X_test, X_train_y, X_test_y = setup()
    print(X_train[0],X_train[0,::2],X_train[0,1::2])
    plt.scatter(X_train[0,::2],X_train[0,1::2])
    task2_1()
    plt.show()
