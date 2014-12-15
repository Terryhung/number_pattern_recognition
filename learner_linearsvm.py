from sklearn import ensemble, datasets, grid_search, svm
import numpy as np

if __name__ == "__main__":
    # data pre
    X_train, y_train = datasets.load_svmlight_file('ml14fall_train.dat', n_features="12810")
    x_test, y_test = datasets.load_svmlight_file('ml14fall_test1_no_answer.dat', n_features="12810")
    y_train = y_train.astype(int)
    for i in range(22, 32):
        y_train[y_train == i] = i-10

    # algorithm: random forest
    svc = svm.LinearSVC()
    parameters = {'C': [0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2'], 'multi_class': ['ovr', 'crammer_singer']}
    clf = grid_search.GridSearchCV(svc, parameters, n_jobs=5)
    clf.fit(X_train.toarray(), y_train)
    print clf.best_params_
    y_predict = clf.best_estimator_.predict(x_test.toarray())
    np.savetxt("svm_linear.txt", y_predict, fmt="%s", newline='\n')
