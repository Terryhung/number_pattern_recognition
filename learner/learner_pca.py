from sklearn import datasets, grid_search, svm, ensemble
import numpy as np

if __name__ == "__main__":

    # data pre
    X_train, y_train = datasets.load_svmlight_file('../datasets/hog_ml14fall_train.dat', n_features="12810")
    x_test, y_test = datasets.load_svmlight_file('../datasets/hog_ml14fall_test1_no_answer.dat', n_features="12810")
    y_test = np.load('../datasets/test_answer.npy')
    y_train = y_train.astype(int)
    print ("finish load")

    # algorithm: random forest
    clf = ensemble.BaggingClassifier(
    base_estimator=svm.LinearSVC(C=10),
    n_estimators=150, oob_score=True, n_jobs=10, verbose=1)
    clf.fit(X_train.toarray(), y_train)
