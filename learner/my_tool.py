import os
import sys

def read_data(filename):
    path = os.path.dirname(__file__)
    rel_path = "../datasets/" + filename
    abs_file_path = os.path.join(path, rel_path)
    return abs_file_path
