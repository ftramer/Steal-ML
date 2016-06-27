__author__ = 'Fan'

import numpy as np
from sklearn.metrics import accuracy_score

class OfflineMethods:
    RT_in_F = 'retrain in F'
    RT_in_X = 'retrain in X'
    SLV_in_F = 'solve in F'


class OfflineBase(object):
    def __init__(self, oracle, X_ex, y_ex, X_test, y_test, n_features):
        self.X_ex = X_ex
        self.y_ex = y_ex
        self.X_test = X_test
        self.y_test = y_test

        self.n_features = n_features
        self.oracle = oracle
        self.clf2 = None

    def set_clf2(self, clf2):
        assert clf2 is not None
        if hasattr(clf2, 'predict'):
            self.clf2 = clf2.predict
        else:
            self.clf2 = clf2

    def do(self):
        pass

    def benchmark(self):
        # L_unif
        assert self.clf2 is not None
        X_unif = np.random.uniform(-1, 1, (1000, self.n_features))

        y_unif_ref = self.oracle(X_unif, count=False)
        y_unif_pred = self.clf2(X_unif)

        y_test_ref = self.oracle(self.X_test, count=False)
        y_test_pred = self.clf2(self.X_test)

        L_unif = 1 - accuracy_score(y_unif_ref, y_unif_pred)
        L_test = 1 - accuracy_score(y_test_ref, y_test_pred)

        if -1 in self.y_test:
            if -1 not in y_test_pred:
                y_test_pred = [y if y == 1 else -1 for y in y_test_pred]
            if -1 not in y_test_ref:
                y_test_ref  = [y if y == 1 else -1 for y in y_test_ref]

        print '------'
        print self.__class__.__name__
        print 'Oracle has a test score of %f' % accuracy_score(self.y_test, y_test_ref)
        print 'Extract has a test score of %f' % accuracy_score(self.y_test, y_test_pred)
        print 'L_unif = %f, L_test = %f' % (L_unif, L_test)
        print '------'

        return L_unif, L_test
