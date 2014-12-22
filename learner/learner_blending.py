from sklearn import ensemble, datasets, svm, linear_model
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
    print ("Finish loading data")

    # algorithm: svm and linear svm
    svc_linear = load_model('linear_svm_bagging')
    svc_kernel = load_model('svm_kernel')
    print ("Finish loading model")
   
   # blending
    y_linear = svc_linear.predict(X_train).astype(int)
    # y_kernel = svc_kernel.predict(X_train.toarray())
    y_kernel = np.loadtxt("svm_pred.txt").astype(int)
    
    # start blending
    print ("start blending model")
    X_blending = np.vstack((y_linear, y_kernel))
    X_blending = X_blending.T
    clf = svm.LinearSVC()
    clf.fit(X_blending, y_train)

    # start blending
    print ("start predict")
    y_linear_test = svc_linear.predict(x_test).astype(int)
    # y_kernel_test = svc_kernel.predict(x_test.toarray())
    y_kernel_test = np.loadtxt("svm_test.txt").astype(int)
    X_blending_test = np.vstack((y_linear_test, y_kernel_test)).T
    
    
    # err output
    print ("start test")
    y_predict = clf.predict(X_blending_test)
    np.savetxt("../answer/blending_hog.txt",
               y_predict, fmt="%s", newline='\n')
