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
    svc = svm.SVC()
    parameters = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 0.1, 0.01], 'kernel': ['rbf']},
        {'C': [1, 10, 100, 1000], 'degree': [2, 3, 4, 5], 'kernel': ['poly']},
    ]
    clf = grid_search.GridSearchCV(svc, parameters, n_jobs=10)
    clf.fit(X_train.toarray(), y_train)
    print clf.best_params_
    y_predict = clf.best_estimator_.predict(x_test.toarray())
    np.savetxt("svm_kernel.txt", y_predict, fmt="%s", newline='\n')
