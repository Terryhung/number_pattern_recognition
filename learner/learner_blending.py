from sklearn import ensemble, datasets, svm, linear_model
import numpy as np
from my_tool import read_data
from my_tool import add_model
from my_tool import load_model

if __name__ == "__main__":
    # data pre
    x_test, y_test = datasets.load_svmlight_file(
        read_data('hog_ml14fall_test1_no_answer.dat'), n_features="12810")
    print ("Finish loading data")

    # algorithm: svm and linear svm and regression
    svc_linear_bagging = load_model('linear_svm_bagging')
    svc_linear = load_model('linear_svm')
    svc_kernel = load_model('svm_kernel')
    svc_kernel_125 = load_model('svm_kernel_125')
    linear_regression = load_model('regression')
    print ("Finish loading model")

    # start blending
    print ("start predict")
    y_kernel = svc_kernel.predict(x_test)
    np.savetxt("./blending_data/kernel.txt",
               y_kernel, fmt="%s", newline='\n')
    y_kernel_125 = svc_kernel_125.predict(x_test)
    np.savetxt("./blending_data/kernel_125.txt",
               y_kernel_125, fmt="%s", newline='\n')
    y_linear_bagging = svc_linear_bagging.predict(x_test)
    np.savetxt("./blending_data/linear_bagging.txt",
               y_linear_bagging, fmt="%s", newline='\n')
    y_linear = svc_linear.predict(x_test)
    np.savetxt("./blending_data/linear.txt",
               y_linear, fmt="%s", newline='\n')
    y_regression = linear_regression.predict(x_test)
    np.savetxt("./blending_data/regression.txt",
               y_regression, fmt="%s", newline='\n')
    print ("Finish predict")

    # err output
    # print ("start test")
    # np.savetxt("../answer/blending_hog.txt",
    #             y, fmt="%s", newline='\n')
