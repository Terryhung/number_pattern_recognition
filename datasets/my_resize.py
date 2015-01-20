from sklearn import datasets
import numpy as np
from scipy import misc
import sys
def photo_resize(photo, height, width):
    # left
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
    try:
        new_photo = photo[left:right, top:down]
    except:
        misc.imsave('t.jpg', photo)
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
    count = 0
    fileopen = open('train_val.txt', 'w')
    for index, data in enumerate(X_train):
        photo = "image_" + str(index) + '.jpg'
        route = 'data/my_photo/' + photo + ' ' + str(y_train[index]) + '\n'
        fileopen.write(route)
