__author__ = 'fan'

import numpy as np
from sklearn.metrics import accuracy_score


class HyperSolver:
    def __init__(self, p=1, n=-1):
        self.w = None
        self.b = None
        self.POS = p
        self.NEG = n

    def fit(self, X, Y):
        m = len(X)
        para = np.matrix(X)
        bb = -1 * np.ones(m).T
        print para, bb

        self.w, _, _, _ = np.linalg.lstsq(para, bb)
        self.b = 1

        yy = self.predict(X)
        score = sum(yy == Y) / float(len(yy))
        print score
        if score < 0.5:
            self.w *= -1
            self.b *= -1

    def predict(self, X):
        yy = np.inner(X, self.w)
        b = self.b * np.ones(yy.shape)
        d = np.sign(np.inner(X, self.w) + b)
        d[d == 1] = self.POS
        d[d == -1] = self.NEG
        return d

    def get_params(self, deep=True):
        return {}

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
