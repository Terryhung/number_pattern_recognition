from sklearn import svm
from sklearn import linear_model
import numpy as np
import scipy as sp
from sklearn import datasets

if __name__ == "__main__":
    X_train, y_train = datasets.load_svmlight_file('ml14fall_train.dat', n_features="12810")
    x_test, y_test = datasets.load_svmlight_file('ml14fall_test1_no_answer.dat', n_features="12810")
    y_train = y_train.astype(int)
    clf = linear_model.LogisticRegression()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(x_test)
    np.savetxt("regression.txt", y_predict, fmt="%s", newline='\n')
