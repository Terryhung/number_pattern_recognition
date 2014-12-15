from sklearn import ensemble
import numpy as np
from sklearn import datasets

if __name__ == "__main__":
    # data pre
    X_train, y_train = datasets.load_svmlight_file('ml14fall_train.dat', n_features="12810")
    x_test, y_test = datasets.load_svmlight_file('ml14fall_test1_no_answer.dat', n_features="12810")
    y_train = y_train.astype(int)

    # algorithm: random forest
    clf = ensemble.RandomForestClassifier()
    clf.fit(X_train.toarray(), y_train)
    y_predict = clf.predict(x_test)
    np.savetxt("regression.txt", y_predict, fmt="%s", newline='\n')
