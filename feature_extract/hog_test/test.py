import matplotlib.pyplot as plt

from scipy import misc
import numpy as np
from skimage.feature import hog
from skimage import data, color, exposure


image_1 = misc.imread('m2.jpg', 1)
image = color.rgb2gray(image_1)

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
print np.count_nonzero(image_1)
print np.count_nonzero(hog_image)
