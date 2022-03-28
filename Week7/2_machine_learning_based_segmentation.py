import numpy as np

import os, sys
import glob

import tensorflow as tf

from tqdm import tqdm

from skimage.io import imread

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, InputLayer
from tensorflow.keras.optimizers import Adam


## Configure the network
# batch_size to traink

batch_size = 20 * 256
# number of output classes
nb_classes = 135
# number of epochs to train
nb_epoch = 400

# number of convolutional filters to use
nb_filters = 20
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 5

model = Sequential([
    InputLayer(input_shape=(29, 29, 1)),
    Conv2D(filters=nb_filters, kernel_size=nb_conv, activation='relu'),
    MaxPool2D(pool_size=(nb_pool, nb_pool)),
    Dropout(0.5),
    Conv2D(filters=nb_filters, kernel_size=nb_conv, activation='relu'),
    MaxPool2D(pool_size=(nb_pool, nb_pool)),
    Dropout(0.25),
    Flatten(),
    Dense(units=4000, activation='relu'),
    Dense(units=nb_classes, activation='softmax'),
])
    
optimizer = Adam(lr=1e-4, epsilon=1e-08)

model.compile(optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])


## Train model - uncoment to perform the training yourself
#

#train = numpy.load('train.npz')
#x_train = train['x_train'].reshape((-1, 29, 29, 1))
#y_train = train['y_train']
#
#early_stopping = EarlyStopping(patience=10)
#history = model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size,
#                    callbacks=[early_stopping], validation_split=0.2)
#model.save_weights('keras.h5')

## Load the pretrained network
model.load_weights('keras.h5') 


# # 2.2
def task2_2():
    test = np.load('test.npz')
    x_test = test['x_test']
    y_test = test['y_test']
    x_test_reshape = test["x_test"].reshape((-1,29,29,1)) # adds depth -> [patches, in_rows, in_cols, depth/channel].

    model.evaluate(x_test_reshape, y_test)

def task2_3(plot=True, debug=False):
    def extract_patches(tensor, shape=(29,29)):
        shaped = tf.expand_dims(tensor, axis=-1) # adds batch = ->[batch, in_rows, in_cols, depth].
        patch = tf.image.extract_patches(images=shaped,
                            sizes=[1, shape[0], shape[1], 1],
                            strides=[1, 1, 1, 1],
                            rates=[1, 1, 1, 1],
                            padding='SAME')
        
        return patch / 255

    img_folder = 'test_images\\image'
    seg_folder = 'test_images\\seg'
    img_path = glob.iglob(f'{img_folder}/*')

    imgs = []
    segs = []
    paths = []
    for img in glob.iglob(f'{img_folder}/*'):
        imgs.append(np.expand_dims(imread(img), axis=0)) # add channel dimension (1 for grayscale)
        paths.append(img)

    for seg in glob.iglob(f'{seg_folder}/*'):
        segs.append(imread(seg))
    
    # Debug mode: use 
    print("Compiling 20 images to patches:")
    if debug:
        x_test = np.array([extract_patches(imgs)])
    else:
        x_test = np.array([extract_patches(img) for img in tqdm(imgs)]) 

    x_test_reshaped = x_test.reshape((-1,29,29,1))

    # Debug mode: use x_test.reshaped[0]
    print("Prediction on patches dataset -> Segmentations:")
    prediction = model.predict(x_test_reshaped, verbose=1)

    new_segmentations = np.argmax(prediction, axis=1).reshape(-1, 256, 256)

    if plot:
        if debug:
            fig, axs = plt.subplots(1, 3, figsize=(16, 8))

            axs[0].imshow(np.squeeze(imgs[0]), cmap='gray')
            axs[1].imshow(segs[0], cmap='gray', vmin=0, vmax=135)
            im = axs[2].imshow(new_segmentations[0], cmap='gray', vmin=0, vmax=134)

            divider = make_axes_locatable(axs[2])
            cax = divider.append_axes('right', size='5%', pad=0.05)

            fig.colorbar(im, cax=cax, orientation='vertical')
        else:
            i = np.random.randint(0,18)
            
            fig, axs = plt.subplots(2, 3, figsize=(16, 8))

            axs[0][0].imshow(np.squeeze(imgs[i]), cmap='gray')
            axs[0][0].set_title("{}".format(paths[i]))
            axs[0][1].set_title("Ground Truth")
            axs[0][2].set_title("Prediction")
            axs[0][1].imshow(segs[i], cmap='gray', vmin=0, vmax=135)
            im = axs[0][2].imshow(new_segmentations[i], cmap='gray', vmin=0, vmax=134)

            divider = make_axes_locatable(axs[0][2])
            cax = divider.append_axes('right', size='5%', pad=0.05)

            fig.colorbar(im, cax=cax, orientation='vertical')

            axs[1][0].imshow(np.squeeze(imgs[i+1]), cmap='gray')
            axs[1][1].imshow(segs[i+1], cmap='gray', vmin=0, vmax=135)
            axs[1][0].set_title("{}".format(paths[i+1]))
            axs[1][1].set_title("Ground Truth")
            axs[1][2].set_title("Prediction")
            im1 = axs[1][2].imshow(new_segmentations[i+1], cmap='gray', vmin=0, vmax=134)

            divider = make_axes_locatable(axs[1][2])
            cax = divider.append_axes('right', size='5%', pad=0.05)

            fig.colorbar(im1, cax=cax, orientation='vertical')
            
        plt.show()

    return new_segmentations, segs

def task2_4(predictions, gt):
    def _dice(gt, pred, i):
        gt = gt.flatten()
        pred = pred.flatten()
        intersection = np.sum(pred[gt==i]) * 2.
        dsc = intersection / (np.sum(pred) + np.sum(gt))
        return dsc

    def dsc_multilabel(gt, pred, n_classes=135):
        dsc = 0
        for i in range(n_classes):
            dsc += _dice(gt, pred, i)
        return dsc / n_classes

if __name__=='__main__':
    #task2_2()
    predictions, ground_truth = task2_3(debug=False)
    print(task2_4(predictions, ground_truth))


    