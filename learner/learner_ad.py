from sklearn import datasets, grid_search, ensemble
import numpy as np

if __name__ == "__main__":
    # data pre
    X_train, y_train = datasets.load_svmlight_file('../datasets/hog_ml14fall_train.dat', n_features="12810")
    x_test, y_test = datasets.load_svmlight_file('../datasets/hog_ml14fall_test1_no_answer.dat', n_features="12810")
    y_train = y_train.astype(int)

    # algorithm: random forest
    svc = ensemble.AdaBoostClassifier()
    parameters = {'n_estimators': range(50, 100, 5)}
    clf = grid_search.GridSearchCV(svc, parameters, n_jobs=10)
    clf.fit(X_train.toarray(), y_train)

    print clf.best_params_
    y_predict = clf.best_estimator_.predict(x_test.toarray())
    np.savetxt("../answer/ad_hog.txt", y_predict, fmt="%s", newline='\n')
