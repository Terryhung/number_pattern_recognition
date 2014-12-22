from sklearn import ensemble, datasets, svm, linear_model
import numpy as np
from my_tool import read_data
from my_tool import add_model
from my_tool import load_model

if __name__ == "__main__":
    # start blending
    print ("start predict")
    y_kernel = np.loadtxt('./blending_data/svm_test.txt')
    y_kernel_125 = np.loadtxt('./blending_data/kernel_125.txt')
    y_linear_bagging = np.loadtxt('./blending_data/linear_bagging.txt')
    y_linear = np.loadtxt('../answer/svm_linear_hog.txt')
    y_regression = np.loadtxt('./blending_data/regression.txt')
    y_forset = np.loadtxt('../answer/forset_hog.txt')
    print ("Finish predict")
    
    y = np.vstack((y_kernel, y_kernel_125, y_linear_bagging, y_regression, y_forset, y_linear)).T
    y_prediction = []
    for i in range(y.shape[0]):
        unique,pos = np.unique(y[i],return_inverse=True)
        counts = np.bincount(pos)
        maxpos = counts.argmax()
        y_prediction.append(int(unique[maxpos]))
        print ("finish"+str(i))
    y_prediction = np.asarray(y_prediction)
    np.savetxt("../answer/blending.txt",
                y_prediction, fmt="%s", newline='\n')

