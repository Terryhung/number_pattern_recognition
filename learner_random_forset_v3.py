from sklearn import ensemble, datasets, grid_search
import numpy as np

if __name__ == "__main__":
    # data pre
    X_train, y_train = datasets.load_svmlight_file('hog_ml14fall_train.dat', n_features="12810", dtype="float64")
    x_test, y_test = datasets.load_svmlight_file('hog_ml14fall_test1_no_answer.dat', n_features="12810", dtype="float64")
    y_train = y_train.astype(int)


    # algorithm: random forest
    random_for = ensemble.RandomForestClassifier()
    parameters = {'n_estimators': [500, 600, 700, 800, 900, 1000], 'max_features': ['log2', 'sqrt'], 'n_jobs': [10]}
    clf = grid_search.GridSearchCV(random_for, parameters)
    clf.fit(X_train.toarray(), y_train)
    print clf.best_params_
    y_predict = clf.best_estimator_.predict(x_test.toarray())
    np.savetxt("forset_v3.txt", y_predict, fmt="%s", newline='\n')
