from sklearn import datasets, grid_search, svm
import numpy as np

if __name__ == "__main__":
    # data pre
    X_train, y_train = datasets.load_svmlight_file('ml14fall_train.dat', n_features="12810")
    X_test, y_test = datasets.load_svmlight_file('ml14fall_test1_no_answer.dat', n_features="12810")
    y_train = y_train.astype(int)

    X_train = X_train.toarray()
    X_test = X_test.toarray()
    y_train = y_train.toarray()
    y_test = y_test.toarray()

    np.save('datasets/X_train', X_train)
    np.save('datasets/y_train', y_train)
    np.save('datasets/X_test', X_test)
    np.save('datasets/y_test', y_test)
