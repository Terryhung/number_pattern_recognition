from __future__ import division
import numpy as np
import scipy as sp
from sklearn import datasets, grid_search, linear_model, ensemble
import my_tool

def cs_err_arr(y_true, y_pred):
    hard_err = (y_true < 12) & (y_true != y_pred)
    err_in_three_cases = ((y_true != y_pred) &
                          (y_true + 10 != y_pred) & (y_true - 10 != y_pred))
    soft_err = ((y_true >= 12) & err_in_three_cases)
    return (hard_err | soft_err)
if __name__ == "__main__":
    # loading data
    X_train, y_train = datasets.load_svmlight_file('../datasets/hog_ml14fall_train.dat', n_features="12810")
    x_test, y_test = datasets.load_svmlight_file('../datasets/hog_ml14fall_test1_no_answer.dat', n_features="12810")
    y_test = np.load('../datasets/test_answer.npy')
    y_train = y_train.astype(int)
    print ("Finish loading")

    # logistic regression
    reg = linear_model.LogisticRegression()
    reg.fit(X_train, y_train)
    y_predict = reg.predict(x_test)
    print cs_err_arr(y_test, y_predict)

    # logistic regression bagging
    clf = ensemble.BaggingClassifier(
    base_estimator=linear_model.LogisticRegression(C=10),
    n_estimators=150, oob_score=True, n_jobs=10, verbose=1)
    clf.fit(X_train.toarray(), y_train)
    y_predict = clf.best_estimator_.predict(x_test)
    print cs_err_arr(y_test, y_predict)
