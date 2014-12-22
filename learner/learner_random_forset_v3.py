from sklearn import ensemble, datasets, grid_search
import numpy as np
from my_tool import read_data
from my_tool import add_model

if __name__ == "__main__":
    # data pre
    X_train, y_train = datasets.load_svmlight_file(read_data('hog_ml14fall_train.dat'), n_features="12810", dtype="float64")
    x_test, y_test = datasets.load_svmlight_file(read_data('hog_ml14fall_test1_no_answer.dat'), n_features="12810", dtype="float64")
    y_train = y_train.astype(int)


    # algorithm: random forest
    clf = ensemble.RandomForestClassifier(n_estimators=870, max_features="sqrt", n_jobs=10)
    clf.fit(X_train.toarray(), y_train)
    y_predict = clf.predict(x_test.toarray())
    np.savetxt("../answer/forset_hog.txt", y_predict, fmt="%s", newline='\n')

    # Save model
    print 'Do you want to save this model? <N | filename>'
    answer = raw_input()
    add_model(answer, clf)
