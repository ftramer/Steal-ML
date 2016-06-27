__author__ = 'Fan'

import copy
import logging

import numpy as np

from sklearn import svm
from sklearn.datasets import load_svmlight_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from algorithms.OnlineBase import OnlineBase


class LordMeek(OnlineBase):
    def __init__(self, target, test_xy, error=None, delta=None):
        self.X_test, self.y_test = test_xy
        super(self.__class__, self).__init__('LM', +1, -1, target, len(self.X_test[0]), 'uniform', error)

        self.e = error
        self.delta = delta

        if 0 in self.y_test:
            self.NEG = 0
        elif -1 in self.y_test:
            self.NEG = -1
        else:
            print 'Watch out for test file! Neither 0 nor 1 is included!'

    def find_starters(self):
        """
        This function finds a pair of instances. One positive and one negative
        :param clf: classifier being extracted
        :return: (x+, x-) a pair of instances
        """
        # perdict = 1 ? inner(x, coef) + intercept_ > 0 : 0

        x_n, x_p = (None, None)
        x_n_found = False
        x_p_found = False
        for d in self.X_test:
            if x_n_found and x_p_found:
                break

            if self.query(d) == 1 and (not x_p_found):
                x_p = d
                x_p_found = True
            elif self.query(d) == self.NEG and (not x_n_found):
                x_n = d
                x_n_found = True
        return x_p, x_n

    def find_witness(self):
        x_p, x_n = self.find_starters()
        assert x_p is not None and self.query(x_p) == 1
        assert x_n is not None and self.query(x_n) == self.NEG
        dim = len(x_p)
        assert dim == len(x_n)

        last_p = -1
        for i in xrange(0, dim):
            # record the old value
            last_x_p_i = x_p[i]
            # change the value
            x_p[i] = x_n[i]
            if self.query(x_p) == self.NEG:
                # if flips
                last_x_p = copy.copy(x_p)
                last_x_p[i] = last_x_p_i
                assert self.query(x_p) == self.NEG and self.query(last_x_p) == 1
                logger.debug('witness found for dim %d' % i)
                return i, last_x_p, x_p

        return None

    def line_search(self, x, i):
        """
        starting at x (a negative point), search along dimension i, find a point very close to boundary
        :param x: starting point
        :param i: dimension to search
        :return: return the point near boundary
        """
        # make sure to start at a negative point
        assert self.query(x) == self.NEG
        # detach
        new_x = copy.copy(x)

        # phase II: binary search between init and x[i]
        def b(l, r):
            # print 'binary search [%f, %f]' % (l, r)
            # c(l) = 1 && c(r) = 0
            m = 0.5 * (l + r)
            new_x[i] = m
            if self.query(new_x) == self.NEG:
                return b(l, m)
            else:
                if abs(l - m) < self.e:
                    return m, abs(l - m)
                return b(m, r)

        # phase I: exponential explore
        init_xi = x[i]
        step = 1.0 / 100

        # TODO not float64 yet
        while new_x[i] < np.finfo('f').max:
            new_x[i] += step
            if self.query(new_x) == 1:
                return b(new_x[i], init_xi)

            new_x[i] = init_xi
            new_x[i] -= step
            if self.query(new_x) == 1:
                return b(new_x[i], init_xi)

            step *= 2

    def do(self):
        f, sp, sn = self.find_witness()
        sp_f = sp[f]
        sn_f = sn[f]

        w_f = 1.0 * (sp_f - sn_f) / abs(sp_f - sn_f)
        x0, _ = self.push_to_b(sn, sp, self.e)

        # get a x1 with gap(x0,x1) = 1 & c(x1) = 0
        x1 = copy.copy(x0)
        x1[f] -= w_f

        u = np.zeros(len(x0))
        w = np.zeros(len(x0))  # target
        w[f] = w_f
        for i in xrange(0, len(x0)):
            if i == f:
                continue
            # unit vector along the ith dimension
            u[i] = 1.0
            a = np.add(x1, u / self.delta)
            b = np.add(x1, -u / self.delta)
            if self.query(a) == self.query(b):
                w[i] = 0
            else:
                logger.debug('Line search for dim %d', i)
                new_x_i, err = self.line_search(x1, i)
                w[i] = 1.0 / (new_x_i - x1[i])
            u[i] = 0.0

        b = self.clf1.intercept_ / self.clf1.coef_[0][f]
        # print w, b
        # print self.clf1.coef_ / self.clf1.coef_[0][f]

        # test
        error_clf = 0.0
        error_lrn = 0.0
        for test_x, test_y in zip(self.X_test, self.y_test):
            t = 1 if np.inner(w, test_x) + b > 0 else self.NEG
            if t != test_y:
                error_lrn += 1
            if self.clf1.predict(test_x) != test_y:
                error_clf += 1

        pe_clf = 1 - error_clf/ len(self.y_test)
        pe_lrn = 1 - error_lrn/ len(self.y_test)

        print 'L_test = %f' % max(pe_clf - pe_lrn, .0)
        print 'L_unif = %f' % (0.0,)


if __name__ == '__main__':

    X_train, y_train = load_svmlight_file('../targets/diabetes/test.scale', n_features=8)
    X_test, y_test = load_svmlight_file('../targets/diabetes/test.scale', n_features=8)
    X_train = X_train.todense().tolist()
    X_test  = X_test.todense().tolist()

    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)

    n_features = len(X_train[0])
    deltas = (1, .1, .01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
    for e in deltas:
        delta = 1.0 / 10000
        print 'error bound=%f' % e
        ex = LordMeek(clf, (X_test, y_test), error=e, delta=delta)
        ex.do()
        print 'nq=%d' % (ex.get_n_query())
