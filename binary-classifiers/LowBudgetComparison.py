__author__ = 'fan'
import matplotlib as mpl
mpl.use('Agg')
import logging
import pickle

from sklearn import svm
from sklearn.datasets import load_svmlight_file
import sklearn.metrics as sm
import numpy as np

from algorithms.RBFTrainer import RBFKernelRetraining
from algorithms.RBFTrainer import OfflineMethods
from utils.benchmark import Benchmark

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

frac_range = np.arange(10, 270, 50) / 100.0
frac_range_in_f = range(50, 270, 50)

log2n_range = range(7, 13, 1)


def run(train_data, test_data, n_features, gamma, C, feature_type='uniform'):
    print train_data

    b = Benchmark(train_data + '-lb')

    X, Y = load_svmlight_file(train_data, n_features=n_features)
    Xt, Yt = load_svmlight_file(test_data, n_features=n_features)
    rbf_svc = svm.SVC(kernel='rbf', gamma=gamma, C=C).fit(X, Y)

    print '--------------- original -----------------'
    baseline = sm.accuracy_score(Yt, rbf_svc.predict(Xt))
    b.add_kv('baseline', baseline)
    print 'original: %f' % baseline

    def retrain_in_x_with_grid():
        print '--------------- retrain in X with grid -----------------'
        bf_pts, bf_test, bf_train, bn_q = [], [], [], []
        for budget in np.linspace(100, 5000, 15):
            ex = RBFKernelRetraining(test_data,
                              rbf_svc.predict, Xt, Yt,
                                     n_features, OfflineMethods.RT_in_X,
                              budget,
                              error=1, kernel='empty', ftype=feature_type)

            s_train, s_test = ex.grid_retrain_in_x(n_pts=-1, exhaust=True)
            bf_pts.append(budget)
            bf_train.append(s_train)
            bf_test.append(s_test)
            bn_q.append(ex.get_n_query())

        b.add('retrain in X with grid (low budget)',
              bf_pts, 'budget',
              (bf_train, bf_test,), ('CV score', 'test score',),
              (bn_q,), ('num. of queries',))

        del bf_pts, bf_train, bf_test, bn_q

    def retrain_in_f_with_grid():
        print '--------------- retrain in F with grid -----------------'
        bn_dim, bn_budget, bf_train, bf_test, bn_q = [], [], [], [], []
        for log2n in log2n_range:
            dim = 2 ** log2n
            for budget in np.linspace(100, 5000, 15):
                print 'RBF DIM      : %d' % dim
                print 'BUDGET       : %d' % budget
                ex = RBFKernelRetraining(test_data,
                                  rbf_svc.predict, Xt, Yt,
                                         n_features, OfflineMethods.RT_in_F,
                                  budget,
                                  error=1, kernel='rbf', fmap=None, ftype=feature_type)
                s_train, s_test = ex.grid_retrain_in_f(dim, frugal=False, exhaust=True)
                q = ex.get_n_query()

                print 'SCORE TRAIN  : %f' % s_train
                print 'SCORE TEST   : %f' % s_test
                print 'Q            : %d' % q
                print ''
                bn_dim.append(dim)
                bn_budget.append(budget)
                bf_train.append(s_train)
                bf_test.append(s_test)
                bn_q.append(q)

        # b.add('retrain in F with grid',
        #       b, 'retrain pts / training pts',
        #       (bf_train, bf_test,), ('CV score', 'test score',),
        #       (bn_q,), ('number of queries',))
        # del bf_pts, bf_test, bf_train, bn_q
        with open(train_data + '-lb-grid.pkl', 'wb') as f:
            pickle.dump((bn_dim, bn_budget, bf_train, bf_test, bn_q), f)

    retrain_in_x_with_grid()
    retrain_in_f_with_grid()
    b.store()


run('data/diabetes.aa', 'data/diabetes.ab', 8, gamma=2.0, C=.5, feature_type='uniform')
# run('data/australian.aa', 'data/australian.ab', 14, gamma=0.03125, C=.125, feature_type='uniform')
run('./data/breast-cancer.aa', './data/breast-cancer.ab', 10, gamma=0.5, C=.125, feature_type='uniform')
# run('./data/cod-rna.aa', './data/cod-rna.ab', 14, gamma=0.5, C=.125, e=1e-3, feature_type='uniform')
run('./data/fourclass.aa', './data/fourclass.ab', 2, gamma=8.0, C=128, feature_type='uniform')
