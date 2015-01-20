import numpy as np
import scipy as sp
from sklearn import datasets, grid_search, linear_model, ensemble
import my_tool

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
    print sum((y_predict - y_test) != 0)/y_test.shape[0]

    # logistic regression bagging
    clf = ensemble.BaggingClassifier(
        base_estimator=linear_model.LogisticRegression,
        n_estimators=100,
        oob_score=True,
        n_jobs=10,
        verbose=1
    )
    y_predict = clf.best_estimator_.predict(x_test)
    print sum((y_predict - y_test) != 0)/y_test.shape[0]
