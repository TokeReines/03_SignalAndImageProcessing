import skimage
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import procrustes


def setup():
    X_train = np.loadtxt('diatom/SIPdiatomsTrain.txt', delimiter = ',')
    X_test = np.loadtxt('diatom/SIPdiatomsTest.txt', delimiter = ',')
    X_train_classes = np.loadtxt('diatom/SIPdiatomsTrain_classes.txt', delimiter = ',')
    X_test_classes = np.loadtxt('diatom/SIPdiatomsTest_classes.txt', delimiter = ',')
    return X_train, X_test, X_train_classes, X_test_classes


def procrustes_tranform(dataset):
    X_train, X_test, X_train_y, X_test_y = setup()

    target_points = np.stack((X_train[0, 0::2], X_train[0, 1::2]), axis=1)
    train_points = np.stack((dataset[0:, 0::2], dataset[0:, 1::2]), axis=2)

    # Procrustes takes a target, and an entry to be transformed
    mtxs = [procrustes(target_points, train_points[i]) for i in range(len(dataset))]

    mtxs = np.array(mtxs, dtype=object)

    # Unpack variables
    mtx1s, mtx2s, disparities = mtxs[:,0], mtxs[:,1], mtxs[:,2]
    # We're interrested in mtx2s for task 2.2, and return as same format and shape as dataset:
    return np.array([mtx2s[i].flatten() for i in range(len(dataset))])


def task2_1(plot=False):
    X_train, X_test, X_train_y, X_test_y = setup()

    target_points = np.stack((X_train[0, ::2], X_train[0, 1::2]), axis=1)

    train_xs = X_train[0:, ::2]
    train_ys = X_train[0:, 1::2]
    
    train_points = np.stack((train_xs, train_ys), axis=2)
    # example usage mtx2s[diatom idx][:,0] all x values for post-procrustes for diatom idx in train_points:
    mtx2s = procrustes_tranform(X_train)
    
    # Standardize target point to illustrate with post-procrustes example diatoms 
    target_standard, _, _ = procrustes(target_points, target_points)

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].scatter(target_points[:,0], target_points[:,1], color='r', marker='D')
        axs[1].scatter(target_points[:,0], target_points[:,1], color='r', marker='D', alpha=.5)
        axs[2].scatter(target_standard[:,0], target_standard[:,1], color='r', marker='D', alpha=.25)
        axs[0].set_title("Target diatom")

        axs[1].scatter(train_xs[1], train_ys[1], color='b', marker='1')
        axs[1].scatter(train_xs[2], train_ys[2], color='g', marker='2')
        axs[1].scatter(train_xs[3], train_ys[3], color='y', marker='3')
        axs[1].set_title("Pre-Procrustes Diatoms Original")

        axs[2].scatter(mtx2s[1][0::2], mtx2s[1][1::2], color='b', marker='1')
        axs[2].scatter(mtx2s[2][0::2], mtx2s[2][1::2], color='g', marker='2')
        axs[2].scatter(mtx2s[3][0::2], mtx2s[3][1::2], color='y', marker='3')
        axs[2].set_title("Post-Procrustes Diatoms")

        fig.suptitle("Procrustes Transformation with target diatom in red")
        fig.tight_layout()

    return 

def task2_2():
    # First fit an RF on train set with no Procrustes transformation
    clf = RandomForestClassifier()
    X_train, X_test, X_train_y, X_test_y = setup()  
    clf.fit(X_train, X_train_y)
    # Predictions on the test set
    y_preds = clf.predict(X_test)
    acc_pre = sum(y_preds==X_test_y)/len(X_test_y)
 
    # Fit another RF on Procrustes transformed entries
    clf2 = RandomForestClassifier()
    X_train_procrustes = procrustes_tranform(X_train)
    X_test_procrusted = procrustes_tranform(X_test)
    clf2.fit(X_train_procrustes, X_train_y)
    # Predictions on the standardised test set
    y_preds = clf2.predict(X_test_procrusted)
    acc_post = sum(y_preds==X_test_y)/len(X_test_y)

    print("Accuracy for kNN on pre-Procrustes: %.4f" %acc_pre)
    print("Accuracy for kNN on post-Procrustes: %.4f" %acc_post)

if __name__ == '__main__':
    task2_1(plot=True)
    task2_2()
    plt.show()
