import pickle

import importlib
import os

import sys

import pdb
def main():

    sys.path.append("/Users/ot44/Development/EthicML/tests/oliver_git_svm/test_svm_module")
    module = importlib.import_module("SVMTWO")
    _class = getattr(module, "SVMEXAMPLE")
    pickle.dump(_class, open( "dumped_pickle.p", "wb" ))


if __name__ == '__main__':
    main()
