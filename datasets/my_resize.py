from sklearn import datasets
import numpy as np
from scipy import misc
import sys
def photo_resize(photo, height, width):
    # left
    left = 0
    right = 0
    top = 0
    down = 0
    for i in range(0, height):
        if np.count_nonzero(photo[i, :]) > 0:
            left = i
            break
    for i in range(0, height):
        if np.count_nonzero(photo[height-1-i, :]) > 0:
            right = height-1-i
            break
    for i in range(0, width):
        if np.count_nonzero(photo[:, i]) > 0:
            top = i
            break
    for i in range(0, width):
        if np.count_nonzero(photo[:, width-1-i]) > 0:
            down = width-1-i
            break
    if left != right and top != down:
        new_photo = photo[left:right, top:down]
    else:
        new_photo = photo
    return new_photo


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print ('Usage: my_resize.py <filename> \n')
        exit()

    filename = sys.argv[1]
    X = []
    X_train, y_train = datasets.load_svmlight_file(filename,
                                                   n_features="12810")
    X_train = X_train.toarray()
    for data in X_train:
        data = data.reshape(122, 105)
        new_photo = photo_resize(data, 122, 105)
        new_photo = misc.imresize(new_photo, (122, 105))
        new_photo = new_photo.reshape(12810,)
        X.append(new_photo)
        print("finish")
    X = np.asarray(X)
    save_name = "new_" + filename
    datasets.dump_svmlight_file(X, y_train, save_name)
