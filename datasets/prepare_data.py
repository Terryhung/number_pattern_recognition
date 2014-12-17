from sklearn import datasets, cross_validation
import numpy as np
import scipy as sp

X_train, y_train = datasets.load_svmlight_file('hog_ml14fall_train.dat', n_features="12810", dtype="float64")
X_test, y_test = datasets.load_svmlight_file('hog_ml14fall_test1_no_answer.dat', n_features="12810", dtype="float64")

X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X_train,
                                                                 y_train,
                                                                 test_size=0.2,
                                                                 random_state=12)

X_train = X_train.toarray()
X_cv = X_cv.toarray()
X_test = X_test.toarray()

np.save('X_hog_train', X_train)
np.save('y_hog_train', y_train)
np.save('X_hog_val', X_cv)
np.save('y_hog_val', y_cv)
np.save('X_hog_test', X_test)
np.save('y_hog_test', y_test)
