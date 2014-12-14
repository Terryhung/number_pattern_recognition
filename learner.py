from sklearn import svm
from sklearn import linear_model
import numpy as np
import scipy as sp
from sklearn import datasets

if __name__ == "__main__":
    X_train, y_train = datasets.load_svmlight_file('ml14fall_train.dat')
    x_test = datasets.load_svmlight_file('ml14fall_test1_no_answer.dat')

    clf = linear_model.LogisticRegression()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(x_test)
    fileopen = open("answer.dat", 'w')
    for i in y_predict:
        fileopen.write(str(int(i)))
        fileopen.write("\n")
    fileopen.close()
