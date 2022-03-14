import skimage
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#from scipy.spatial import procrustes


def setup():
    X_train = np.loadtxt('diatom/SIPdiatomsTrain.txt', delimiter = ',')
    X_test = np.loadtxt('diatom/SIPdiatomsTest.txt', delimiter = ',')
    X_train_classes = np.loadtxt('diatom/SIPdiatomsTrain_classes.txt', delimiter = ',')
    X_test_classes = np.loadtxt('diatom/SIPdiatomsTest_classes.txt', delimiter = ',')
    return X_train, X_test, X_train_classes, X_test_classes

def procrustes(target, train):
    target_pro = np.copy(target)
    train_pro = np.copy(train)

    # Translational Alignment
    target_pro -= np.mean(target_pro, 0)
    train_pro -= np.mean(train_pro, 0)
    
    target_pro /= np.linalg.norm(target_pro)
    train_pro /= np.linalg.norm(train_pro)
    
    # Rotational Alignment
    U, s, V = np.linalg.svd(np.dot(train_pro.T, target_pro))
    R = np.dot(U, V.T)
    train_pro = np.dot(train_pro, R) 

    # Scaling Alignment
    train_pro = (train_pro * s.sum())
    return target_pro, train_pro


def procrustes_tranform(dataset):
    X_train, _, _, _  = setup()

    target_points = np.stack((X_train[0, 0::2], X_train[0, 1::2]), axis=1)
    train_points = np.stack((dataset[0:, 0::2], dataset[0:, 1::2]), axis=2)

    # Procrustes takes a target, and an entry to be transformed
    pro = [procrustes(target_points, train_points[i]) for i in range(len(dataset))]

    pro = np.array(pro, dtype=object)

    # Unpack variables
    target_pro, train_pro = pro[:,0], pro[:,1]

    # We're interrested in train_pro for task 2.2, and return as same format and shape as dataset:
    return np.array([train_pro[i].flatten() for i in range(len(dataset))])


def task2_1(plot=False):
    X_train, X_test, X_train_y, X_test_y = setup()

    target_points = np.stack((X_train[0, ::2], X_train[0, 1::2]), axis=1)

    train_xs = X_train[0:, ::2]
    train_ys = X_train[0:, 1::2]

    # example usage pro[diatom idx][:,0] all x values for post-procrustes for diatom idx in train_points:
    train_pro = procrustes_tranform(X_train)
    
    # Standardize target point to illustrate with post-procrustes example diatoms 
    target_standard, _ = procrustes(target_points, target_points)

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

        axs[2].scatter(train_pro[1][0::2], train_pro[1][1::2], color='b', marker='1')
        axs[2].scatter(train_pro[2][0::2], train_pro[2][1::2], color='g', marker='2')
        axs[2].scatter(train_pro[3][0::2], train_pro[3][1::2], color='y', marker='3')
        axs[2].set_title("Post-Procrustes Diatoms")

        fig.suptitle("Procrustes Transformation with target diatom in red")
        fig.tight_layout()

    return 

def task2_2():
    # First fit an RF on train set with no Procrustes transformation
    clf = RandomForestClassifier(n_estimators=180)
    X_train, X_test, X_train_y, X_test_y = setup()  
    clf.fit(X_train, X_train_y)
    # Predictions on the test set
    y_preds = clf.predict(X_test)
    acc_pre = clf.score(X_test, X_test_y)
 
    # Fit another RF on Procrustes transformed entries
    clf2 = RandomForestClassifier(n_estimators=180)
    X_train_procrustes = procrustes_tranform(X_train)
    X_test_procrusted = procrustes_tranform(X_test)
    clf2.fit(X_train_procrustes, X_train_y)
    # Predictions on the standardised test set
    y_preds = clf2.predict(X_test_procrusted)
    acc_post = clf2.score(X_test_procrusted, X_test_y)
    
    print("Accuracy for RFC on pre-Procrustes: %.4f" %acc_pre)
    print("Accuracy for RFC on post-Procrustes: %.4f" %acc_post)

if __name__ == '__main__':
    task2_1(plot=True)
    task2_2()
    plt.show()
