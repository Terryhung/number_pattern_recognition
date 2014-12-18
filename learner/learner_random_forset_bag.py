from sklearn import ensemble, datasets, grid_search
import numpy as np
from my_tool import read_data

if __name__ == "__main__":
    # data pre
    X_train, y_train = datasets.load_svmlight_file(read_data('hog_ml14fall_train.dat'), n_features="12810", dtype="float64")
    x_test, y_test = datasets.load_svmlight_file(read_data('hog_ml14fall_test1_no_answer.dat'), n_features="12810", dtype="float64")
    y_train = y_train.astype(int)


    # algorithm: random forest
    random_for = ensemble.RandomForestClassifier(n_estimators=870, max_features="sqrt", n_jobs=10)
    bagging = ensemble.BaggingClassifier()
    parameters = {'n_estimators': range(100, 1000, 15), 'base_estimator': [random_for]}
    clf = grid_search.GridSearchCV(bagging, parameters, n_jobs=15)
    clf.fit(X_train.toarray(), y_train)
    print clf.best_params_
    y_predict = clf.best_estimator_.predict(x_test.toarray())
    np.savetxt("forset_hog_bag.txt", y_predict, fmt="%s", newline='\n')
