__author__ = 'Fan'

import pickle
import os
import sys

import numpy as np
from sklearn.datasets import load_svmlight_file
import sklearn.metrics as sm
from scipy.special import expit
from scipy.special import logit

from algorithms.awsOnline import AWSOnline

online = False
if online:
    val_name = map(lambda x: 'v%d' % x, range(1, 11))
    print val_name

    test = AWSOnline('ml-lkYRYeldcrH', 1, 0, 10, val_name, 'uniform', .1)
    # spec = FeatureSpec('norm', (-1, 1), (-.85, -.85, -.20, -.51, -.49, -.58, -.48, -.38, -.44, -.55))
    spec = None
    test.collect_with_score(20, spec=spec)
    sys.exit(-1)


class LRSolver(object):
    def __init__(self):
        self.w = None

    def fit(self, X, s):
        _x = np.ones((X.shape[0], X.shape[1] + 1))
        _x[:, : - 1] = X
        self.w, _, _, _ = np.linalg.lstsq(_x, logit(s))

    def predict(self, x):
        _x = np.ones((x.shape[0], x.shape[1] + 1))
        _x[:, : - 1] = x
        score = expit(np.inner(self.w, _x))
        signs = np.sign(score - .5)
        return [0 if x == -1 else 1 for x in signs]

    def score(self, x):
        _x = np.ones((x.shape[0], x.shape[1] + 1))
        _x[:, : - 1] = x
        return np.divide(1, 1 + np.exp(- np.inner(self.w, _x)))


with open('queries_with_score-50', 'rb') as infile:
    a = pickle.load(infile)

X = np.array([x[0] for x in a])
s = np.array([x[2][0] for x in a])
idx = np.random.choice(np.arange(len(X)), 11, replace=False)
_X = X[idx]
_s = s[idx]

ex = LRSolver()
ex.fit(_X, _s)

test_x, test_y = load_svmlight_file(os.getenv('HOME') +
                                    '/Dropbox/Projects/SVM/dataset/breast-cancer/bc-test', n_features=10)
test_x = test_x.todense()

# print ex.predict(test_x)
# print test_y
print sm.accuracy_score(ex.predict(test_x), test_y)
