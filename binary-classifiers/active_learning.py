__author__ = 'Fan'

"""
This corresponds to experiments -> With Kernel Priori -> RBF
"""

import logging
import os
from sklearn import svm
from sklearn.datasets import load_svmlight_file
import sklearn.metrics as sm
import numpy as np

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from algorithms.OnlineBase import OnlineBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

frac_range = np.arange(10, 270, 50) / 100.0
frac_range_in_f = range(50, 270, 50)

log2n_range = range(7, 13, 1)


def CAL_v(name, label_p, label_n, oracle, n_features, ftype, test_x, test_y):
    online = OnlineBase(name, label_p, label_n, oracle, n_features, ftype, error=.5)
    x, y = online.collect_pts(100, -1)
    i = 0
    q = online.get_n_query()
    C_range = np.logspace(-2, 5, 10, base=10)
    gamma_range = np.logspace(-5, 1, 10, base=10)
    param_grid = dict(gamma=gamma_range, C=C_range)
    while q < 3500:
        i += 1
        # h_ = ex.fit(x, y)

        cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv, verbose=0, n_jobs=-1)
        grid.fit(x, y)
        h_ = grid.best_estimator_

        online_ = OnlineBase('', label_p, label_n, h_.predict, n_features, ftype, error=.1)
        x_, _ = online_.collect_pts(10, 200)
        if x_ is not None and len(x_) > 0:
            x.extend(x_)
            y.extend(oracle(x_))
        q += online_.get_n_query()
        pred_y = h_.predict(test_x)
        print len(x), q, sm.accuracy_score(test_y, pred_y)


def CAL(name, label_p, label_n, oracle, n_features, ftype, test_x, test_y):
    online = OnlineBase(name, label_p, label_n, oracle, n_features, ftype, error=.5)

    q = online.get_n_query()
    C_range = np.logspace(-2, 5, 10, base=10)
    gamma_range = np.logspace(-5, 1, 10, base=10)
    param_grid = dict(gamma=gamma_range, C=C_range)

    x, y = online.collect_pts(100, -1)

    i = 0

    cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv, verbose=0, n_jobs=-1)
    grid.fit(x, y)
    h_ = grid.best_estimator_
    while q < 3500:
        i += 1
        # h_ = ex.fit(x, y)

        online_ = OnlineBase('', label_p, label_n, h_.predict, n_features, ftype, error=.1)
        x_ = online_.collect_one_pair()
        if x_ is not None and len(x_) > 0:
            for _x in x_:
                x.append(_x)
                y.append(1)
                cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
                grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv, verbose=0, n_jobs=-1)
                grid.fit(x, y)
                h1 = grid.best_estimator_
                s1 = sm.accuracy_score(y, h1.predict(x))

                y[-1] = -1
                cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
                grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv, verbose=0, n_jobs=-1)
                grid.fit(x, y)
                h2 = grid.best_estimator_
                s2 = sm.accuracy_score(y, h2.predict(x))
                if s1 >= .99 and s2 >= .99:
                    print 'branch 1'
                    y[-1] = oracle(x_)[0]
                elif s1 >= .99 and s2 < .99:
                    print 'branch 2'
                    y[-1] = 1
                elif s1 < .99 and s2 >= .99:
                    print 'branch 3'
                    y[-1] = -1
                else:
                    print 'branch 4: ', s1, s2
                    del x[-1]
                    del y[-1]
                    continue

            if y[-1] == 1:
                h_ = h1
            else:
                h_ = h2

        q += online_.get_n_query()
        pred_y = h_.predict(test_x)
        print q, sm.accuracy_score(test_y, pred_y)


def run(train_data, test_data, n_features, labels, gamma, C, feature_type='uniform'):
    print train_data
    assert os.path.isfile(train_data), '%s is not a file' % train_data
    assert os.path.isfile(test_data), '%s is not a file' % test_data

    X, Y = load_svmlight_file(train_data, n_features=n_features)
    Xt, Yt = load_svmlight_file(test_data, n_features=n_features)
    Xt = Xt.todense()

    if gamma is None:
        gamma = 1.0 / n_features

    if C is None:
        C = 1

    rbf_svc = svm.SVC(kernel='rbf', gamma=gamma, C=C).fit(X, Y)

    print '--------------- original -----------------'
    baseline = sm.accuracy_score(Yt, rbf_svc.predict(Xt))
    print 'original: %f' % baseline

    CAL_v(train_data, labels[1], labels[0], rbf_svc.predict, n_features, feature_type, Xt, Yt)


# run('data/diabetes.aa', 'data/diabetes.ab', 8, (+1, -1), gamma=2.0, C=.5, feature_type='uniform')
# run('data/breast-cancer.aa', 'data/breast-cancer.ab', 10, (1, 0), gamma=0.5, C=.125, feature_type='uniform')
# run('data/australian.aa', 'data/australian.ab', 14, gamma=0.03125, C=.125, feature_type='uniform')
# run('./data/fourclass.aa', './data/fourclass.ab', 2, (1, -1), gamma=8.0, C=128, feature_type='uniform')

run(os.getenv('HOME') + '/Dropbox/Projects/SVM/mushrooms/train.scale',
    os.getenv('HOME') + '/Dropbox/Projects/SVM/mushrooms/test.scale', 112, (1, -1),
    gamma=0.0078125, C=32, feature_type='uniform')
