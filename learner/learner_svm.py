from sklearn import ensemble, datasets, grid_search, svm
import numpy as np
from my_tool import read_data
from my_tool import add_model

if __name__ == "__main__":
    # data pre
    X_train, y_train = datasets.load_svmlight_file(read_data('hog_new_ml14fall_train.dat'), n_features="12810")
    x_test, y_test = datasets.load_svmlight_file(read_data('hog_new_ml14fall_test1_no_answer.dat'), n_features="12810")
    y_train = y_train.astype(int)

    # algorithm: svm
    svc = svm.SVC()
    parameters = [
        {'C': [100],
         'gamma': [0.12],
         'kernel': ['rbf']},
    ]
    clf = grid_search.GridSearchCV(svc, parameters, n_jobs=10)
    clf.fit(X_train.toarray(), y_train)

    print clf.best_params_
    y_predict = clf.best_estimator_.predict(x_test.toarray())
    np.savetxt("../answer/svm_kernel_new_hog.txt",
               y_predict, fmt="%s", newline='\n')

    # Save model
    print 'Do you want to save this model? <N | filename>'
    answer = raw_input()
    add_model(answer, clf)
