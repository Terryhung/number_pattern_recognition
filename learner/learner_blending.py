from sklearn import ensemble, datasets, svm
import numpy as np
from my_tool import read_data
from my_tool import add_model
from my_tool import load_model

if __name__ == "__main__":
    # data pre
    X_train, y_train = datasets.load_svmlight_file(
        read_data('hog_ml14fall_train.dat'), n_features="12810")
    x_test, y_test = datasets.load_svmlight_file(
        read_data('hog_ml14fall_test1_no_answer.dat'), n_features="12810")
    y_train = y_train.astype(int)

    # algorithm: svm and linear svm
    svc_linear = load_model('bagging_svm_100')
    svc_kernel = load_model('svm_kernel')

    # blending
    y_linear = svc_linear.predict(X_train)
    y_kernel = svc_kernel.predict(X_train)

    # start blending
    X_blending = np.vstack((y_linear, y_kernel)).T
    clf = svm.LinearSVC(X_blending, y_train)

    # err output
    y_predict = clf.predict(x_test.toarray())
    np.savetxt("../answer/blending_hog.txt",
               y_predict, fmt="%s", newline='\n')

    # Save model
    print 'Do you want to save this model? <N | filename>'
    answer = raw_input()
    add_model(answer, clf)
