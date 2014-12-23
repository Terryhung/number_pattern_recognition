from sklearn import datasets, grid_search, ensemble
import numpy as np

def err_arr(y_true, y_pred):
    return (y_true != y_pred).mean()
if __name__ == "__main__":
    # data pre
    X_train_all, y_train_all = datasets.load_svmlight_file('../datasets/hog_ml14fall_train.dat', n_features="12810")
    x_test, y_test = datasets.load_svmlight_file('../datasets/hog_ml14fall_test1_no_answer.dat', n_features="12810")
    y_train_all = y_train_all.astype(int)
   
    X_train = np.load('../datasets/X_hog_train.npy')
    y_train = np.load('../datasets/y_hog_train.npy')
    X_val = np.load('../datasets/X_hog_val.npy')
    y_val = np.load('../datasets/y_hog_val.npy')
    # algorithm: random forest
    svc = ensemble.AdaBoostClassifier()
    parameters = {'n_estimators': [500]}
    clf = grid_search.GridSearchCV(svc, parameters, n_jobs=10)
    clf.fit(X_train, y_train)

    print clf.best_params_
    y_predict = clf.best_estimator_.predict(X_val)
    y_pred = clf.best_estimator_.predict(X_train)

    print 'Ein'
    print err_arr(y_pred, y_train).mean()
    print 'Eout'
    print err_arr(y_val, y_predict).mean()

    y_predict = clf.best_estimator_.predict(X_train_all, y_train_all)
    
    np.savetxt("../answer/ad_hog.txt",
                y_predict, fmt="%s", newline='\n')

