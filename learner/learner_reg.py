import numpy as np
import scipy as sp
from sklearn import datasets, grid_search, linear_model
import my_tool

if __name__ == "__main__":
    # loading data
    X_train, y_train = datasets.load_svmlight_file('../datasets/hog_ml14fall_train.dat', n_features="12810")
    x_test, y_test = datasets.load_svmlight_file('../datasets/hog_ml14fall_test1_no_answer.dat', n_features="12810")
    y_train = y_train.astype(int)
    print ("Finish loading")

    # gird search
    reg = linear_model.LogisticRegression()
    parameters = {'C': [0.01, 0.1, 1, 10]}
    clf = grid_search.GridSearchCV(reg, parameters, n_jobs=10)
    clf.fit(X_train, y_train)

    # save data and model
    y_predict = clf.best_estimator_.predict(x_test)
    np.savetxt("../answe/regression.txt", y_predict, fmt="%s", newline='\n')

    my_tool.add_model("regression", clf.best_estimator_)
