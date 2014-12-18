import numpy as np
from sklearn import linear_model as lm, grid_search as gs, svm
import dill
import os
import errno


def err_arr(y_true, y_pred):
    return (y_true != y_pred).mean()


def cs_err_arr(y_true, y_pred):
    hard_err = (y_true < 12) & (y_true != y_pred)
    err_in_three_cases = ((y_true != y_pred) &
                          (y_true + 10 != y_pred) & (y_true - 10 != y_pred))
    soft_err = ((y_true >= 12) & err_in_three_cases)
    return (hard_err | soft_err)


def scoring(estimator, X, y):
    return (1 - cs_err_arr(y, estimator.predict(X)).mean())

X_train = np.load('datasets/X_hog_train.npy')
y_train = np.load('datasets/y_hog_train.npy')
X_val = np.load('datasets/X_hog_val.npy')
y_val = np.load('datasets/y_hog_val.npy')

print 'Finish loading data ...'

clf = gs.GridSearchCV(
    svm.LinearSVC(), {'C': [0.1, 3, 5, 10, 15, 20, 30, 50, 100]},
    n_jobs=20, verbose=2)
clf.fit(X_train, y_train)

print clf.best_estimator_

y_pred = clf.predict(X_train)

print 'Ein:'
print err_arr(y_train, y_pred).mean()
print cs_err_arr(y_train, y_pred).mean()

y_pred = clf.predict(X_val)

print 'Eout:'
print err_arr(y_val, y_pred).mean()
print cs_err_arr(y_val, y_pred).mean()

print 'Do you want to save this model? <N | filename>'

answer = raw_input()

if answer != 'n' and answer != 'N':
    try:
        os.makedirs('models')
    except OSError, exc:
        if exc.errno == errno.EEXIST and os.path.isdir('models'):
            pass
        else:
            raise

    with open(os.path.join('models', answer), 'w') as f:
        f.write(dill.dumps(clf.best_estimator_))
        print 'Model written'
