import os
import errno
import dill


def read_data(filename):
    path = os.path.dirname(__file__)
    rel_path = "../datasets/" + filename
    abs_file_path = os.path.join(path, rel_path)
    return abs_file_path


def save_data(filename):
    path = os.path.dirname(__file__)
    rel_path = "../answer/" + filename
    abs_file_path = os.path.join(path, rel_path)
    return abs_file_path


def add_model(answer, clf):
    if answer != 'n' and answer != 'N':
        try:
            os.makedirs('../models')
        except OSError, exc:
            if exc.errno == errno.EEXIST and os.path.isdir('../models'):
                pass
            else:
                raise

        with open(os.path.join('../models', answer), 'w') as f:
            f.write(dill.dumps(clf))
            print 'Model written'


def load_model(model_name):
    print 'Loading model {0}'.format(model_name)

    with open(os.path.join('../models', model_name), 'r') as f:
        encoded_str = f.read()

    return dill.loads(encoded_str)
