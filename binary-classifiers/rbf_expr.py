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

from algorithms.OnlineBase import OnlineBase
from algorithms.RBFTrainer import RBFKernelRetraining

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

frac_range = np.arange(10, 270, 50) / 100.0
frac_range_in_f = range(50, 270, 50)

log2n_range = range(7, 13, 1)


def retrain_in_x_with_grid(name, label_p, label_n, orable, n_features, ftype, test_x, test_y, benchmark):
    print '--------------- retrain in X with grid -----------------'
    for QSV in xrange(50, 601, 50):
        online = OnlineBase(name, label_p, label_n, orable, n_features, ftype, error=.1)
        online.collect_pts(QSV, -1)
        ex = RBFKernelRetraining(name,
                                 online.get_QSV(), online.get_QSV_labels(),  # training data
                                 online.get_QSV(), online.get_QSV_labels(),  # validation data
                                 test_x, test_y,  # test data
                                 n_features)

        print 'nQSV=%d, Q=%d, ' % (QSV, online.get_n_query()), ex.grid_retrain_in_x()


def retrain_in_f_with_grid(name, label_p, label_n, oracle, n_features, ftype, test_x, test_y, benchmark):
    print '--------------- retrain in F with grid -----------------'
    for n_pts in xrange(50, 601, 50):

        online = OnlineBase(name, label_p, label_n, oracle, n_features, ftype, error=.1)
        online.collect_pts(n_pts, -1)
        ex = RBFKernelRetraining(name,
                                 online.get_QSV(), online.get_QSV_labels(),  # training data
                                 online.get_QSV(), online.get_QSV_labels(),  # validation data
                                 test_x, test_y,  # test data
                                 n_features)

        print 'nQSV=%d, Q=%d, dim=100,' % (n_pts, online.get_n_query()), ex.grid_retrain_in_f(100)


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

    retrain_in_x_with_grid(train_data, labels[1], labels[0], rbf_svc.predict, n_features, feature_type, Xt, Yt, None)
    retrain_in_f_with_grid(train_data, labels[1], labels[0], rbf_svc.predict, n_features, feature_type, Xt, Yt, None)


# run('data/diabetes.aa', 'data/diabetes.ab', 8, (+1, -1), gamma=2.0, C=.5, feature_type='uniform')
# run('data/breast-cancer.aa', 'data/breast-cancer.ab', 10, (1, 0), gamma=0.5, C=.125, feature_type='uniform')
# run('data/australian.aa', 'data/australian.ab', 14, gamma=0.03125, C=.125, feature_type='uniform')
# run('./data/fourclass.aa', './data/fourclass.ab', 2, (1, -1), gamma=8.0, C=128, feature_type='uniform')

run('/home/fanz/Dropbox/Projects/SVM/mushrooms/train.scale',
    '/home/fanz/Dropbox/Projects/SVM/mushrooms/test.scale', 112, (1, -1),
    gamma=0.0078125, C=32, feature_type='uniform')
