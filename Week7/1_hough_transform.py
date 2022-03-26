import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage import feature
from skimage.feature import peak_local_max
from skimage.io import imread


def line_hough_transform(image, theta_res: int = 1, rho_res: float = 1):
    """
    Image space dots are lines in parameterspace. Dots can be edges in a binary image constructed by edge detection.

    Cartesian line: y = a * x + b
        a = gradient or slope
        b = y-translation
    Where (x, y) defines the dot in image space

    Parameter space line:  b = -a * x + y
    Where (a, b) defines the line in parameterspace.

    Since -inf <= a <= inf (we cannot detect vertical lines) we use a discretised version of the normal line:
        p = x * cos(theta) - y * sin(theta)
    Where 0 <= theta <= pi, and p <= image diagonal

    Args:
        rho_res:
        theta_res:
        image: Binary image with edges from edge detection. As a numpy array

    Returns:
        Hough Transform image with sinusoidal curves to infer lines in the input image from.
    """

    rho_max = math.hypot(*image.shape)
    rhos = np.arange(-rho_max // 2, rho_max // 2, rho_res)

    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    thetas_is = np.arange(0, len(thetas))

    hough_space = np.zeros((len(rhos), len(thetas)))

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x, y] == 0:
                continue

            rhos_is = np.rint(x * np.cos(thetas) - y * np.sin(thetas)).astype('int32')
            hough_space[rhos_is, thetas_is] += 1

    return hough_space, rhos, thetas

def hough_lines(hough_image, rhos, thetas, num_peaks=2):
    peaks = peak_local_max(hough_image, num_peaks=num_peaks)
    lines = []
    for rho_i, theta_i in peaks:
        theta = thetas[theta_i]
        rho = rhos[rho_i]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * -b)
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * -b)
        y2 = int(y0 - 1000 * a)
        lines.append(([x1, x2], [y1, y2]))
    return lines




########################################### HOUGH LINES FROM SCRATCH USING NUMPY
# Step 1: The Hough transform needs a binary edges images.  For this particular
# python file, I used the openCV built in Class Canny to create this edge image
# from the original shapes.png file.

# This is the function that will build the Hough Accumulator for the given image
def hough_lines_acc(img, rho_resolution=1, theta_resolution=1):
    ''' A function for creating a Hough Accumulator for lines in an image. '''
    height, width = img.shape  # we need heigth and width to calculate the diag
    img_diagonal = np.ceil(np.sqrt(height ** 2 + width ** 2))  # a**2 + b**2 = c**2
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))

    # create the empty Hough Accumulator with dimensions equal to the size of
    # rhos and thetas
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)  # find all edge (nonzero) pixel indexes

    for i in range(len(x_idxs)):  # cycle through edge points
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)):  # cycle through thetas and calc rho
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1

    return H, rhos, thetas


# This is a simple peaks function that just finds the indicies of the number
# of maximum values equal to num_peaks.  You have to be careful here though, if
# there's any noise in the image it will like create a 'pocket' of local maxima
# values.  This function ignores this and in turn has the tendancy to return
# multiple lines along an actual line in the image.
def hough_simple_peaks(H, num_peaks):
    ''' A function that returns the number of indicies = num_peaks of the
        accumulator array H that correspond to local maxima. '''
    indices = np.argpartition(H.flatten(), -2)[-num_peaks:]
    return np.vstack(np.unravel_index(indices, H.shape)).T


# This more advance Hough peaks funciton has threshold and nhood_size arguments
# threshold will threshold the peak values to be above this value if supplied,
# where as nhood_size will surpress the surrounding pixels centered around
# the local maximum after that value has been assigned as a peak.  This will
# force the algorithm to look eslwhere after it's already selected a point from
# a 'pocket' of local maxima.
def hough_peaks(H, num_peaks, threshold=0, nhood_size=3):
    ''' A function that returns the indicies of the accumulator array H that
        correspond to a local maxima.  If threshold is active all values less
        than this value will be ignored, if neighborhood_size is greater than
        (1, 1) this number of indicies around the maximum will be surpessed. '''
    # loop through number of peaks to identify
    indicies = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1)  # find argmax in flattened array
        H1_idx = np.unravel_index(idx, H1.shape)  # remap to shape of H
        indicies.append(H1_idx)

        # surpess indicies in neighborhood
        idx_y, idx_x = H1_idx  # first separate x, y indexes from argmax(H)
        # if idx_x is too close to the edges choose appropriate values
        if (idx_x - (nhood_size / 2)) < 0:
            min_x = 0
        else:
            min_x = idx_x - (nhood_size / 2)
        if ((idx_x + (nhood_size / 2) + 1) > H.shape[1]):
            max_x = H.shape[1]
        else:
            max_x = idx_x + (nhood_size / 2) + 1

        # if idx_y is too close to the edges choose appropriate values
        if (idx_y - (nhood_size / 2)) < 0:
            min_y = 0
        else:
            min_y = idx_y - (nhood_size / 2)
        if ((idx_y + (nhood_size / 2) + 1) > H.shape[0]):
            max_y = H.shape[0]
        else:
            max_y = idx_y + (nhood_size / 2) + 1

        min_x = int(min_x)
        max_x = int(max_x)
        min_y = int(min_y)
        max_y = int(max_y)
        # bound each index by the neighborhood size and set all values to 0
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                # remove neighborhoods in H1
                H1[y, x] = 0

                # highlight peaks in original H
                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255

    # return the indicies and the original Hough space with selected points
    return indicies, H


# a simple funciton used to plot a Hough Accumulator
def plot_hough_acc(H, plot_title='Hough Accumulator Plot'):
    ''' A function that plot a Hough Space using Matplotlib. '''
    fig = plt.figure(figsize=(10, 10))
    fig.canvas.set_window_title(plot_title)

    plt.imshow(H, cmap='jet')

    plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
    plt.tight_layout()
    plt.show()


# drawing the lines from the Hough Accumulatorlines using OpevCV cv2.line
def hough_lines_draw(img, indicies, rhos, thetas):
    ''' A function that takes indicies a rhos table and thetas table and draws
        lines on the input images that correspond to these values. '''
    for i in range(len(indicies)):
        # reverse engineer lines from rhos and thetas
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 100 * (-b))
        y1 = int(y0 + 100 * (a))
        x2 = int(x0 - 100 * (-b))
        y2 = int(y0 - 100 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)






if __name__ == '__main__':
    # shapes = cv2.imread("Week7/images/cross.png")
    #
    # # read in shapes image and convert to grayscale
    # #shapes = cv2.imread('images/shapes.png')
    # cv2.imshow('Original Image', shapes)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # shapes_grayscale = cv2.cvtColor(shapes, cv2.COLOR_RGB2GRAY)
    #
    # # blur image (this will help clean up noise for Canny Edge Detection)
    # # see Chapter 2.0 for Guassian Blur or check OpenCV documentation
    # shapes_blurred = cv2.GaussianBlur(shapes_grayscale, (5, 5), 1.5)
    #
    # # find Canny Edges and show resulting image
    # canny_edges = cv2.Canny(shapes_blurred, 100, 200)
    # cv2.imshow('Canny Edges', canny_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # run hough_lines_accumulator on the shapes canny_edges image
    # H, rhos, thetas = hough_lines_acc(canny_edges)
    # indicies, H = hough_peaks(H, 3, nhood_size=11)  # find peaks
    # plot_hough_acc(H)  # plot hough space, brighter spots have higher votes
    # hough_lines_draw(shapes, indicies, rhos, thetas)
    # # Show image with manual Hough Transform Lines
    # cv2.imshow('Major Lines: Manual Hough Transform', shapes)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




    image = imread("Week7/images/cross.png", as_gray=True)
    sigma = 5
    low_threshold = 0.8
    high_threshold = 0.9
    gauss_image = gaussian_filter(image,
                                  sigma=sigma)
    canny_image = feature.canny(gauss_image,
                                sigma=sigma,
                                low_threshold=low_threshold,
                                high_threshold=high_threshold,
                                use_quantiles=True)

    hough_image, rhos, thetas = line_hough_transform(image)
    lines = hough_lines(hough_image, rhos, thetas)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].imshow(image, cmap='gray')
    colors = ['r', 'b', 'green']
    for i, line in enumerate(lines):
        # axs[0].scatter(line[0], line[1], c=colors[i])
        axs[0].plot(line[0], line[1], c=colors[i])
    axs[1].imshow(hough_image, cmap='inferno')
    axs[1].set_xlabel('theta')
    axs[1].set_ylabel('rho')
    plt.show()