import skimage
import matplotlib.pyplot as plt
import torch 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import procrustes

def setup():
    X_train = np.loadtxt('diatoms/SIPdiatomsTrain.txt', delimiter = ',')
    X_test = np.loadtxt('diatoms/SIPdiatomsTest.txt', delimiter = ',')
    X_train_classes = np.loadtxt('diatoms/SIPdiatomsTrain_classes.txt', delimiter = ',')
    X_test_classes = np.loadtxt('diatoms/SIPdiatomsTest_classes.txt', delimiter = ',')
    return X_train, X_test, X_train_classes, X_test_classes

def task2_1():
    pass

if __name__ == '__main__':
    X_train, X_test, X_train_y, X_test_y = setup()