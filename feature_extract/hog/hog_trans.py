from sklearn import datasets
import numpy as np
from skimage.feature import hog
import sys

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print ('Usage: hog_trans.py <filename> \n')
        exit()

    filename = sys.argv[1]
    X = []
    X_train, y_train = datasets.load_svmlight_file(filename,
                                                   n_features="12810")
    X_train = X_train.toarray()
    for data in X_train:
        print("finish")
        data = data.reshape(122, 105)
        fd, hog_image = hog(data, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualise=True)
        hog_image = hog_image.reshape(12810,)
        X.append(hog_image)

    X = np.asarray(X)
    save_name = "hog_" + filename
    datasets.dump_svmlight_file(X, y_train, save_name, )
