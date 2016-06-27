__author__ = 'fan'

import os

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn import pipeline, svm
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from matplotlib.colors import Normalize
import sklearn.metrics as sm
from sklearn.datasets import load_svmlight_file

from HyperSolver import HyperSolver
from OnlineBase import OnlineBase


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class GridRBFSolver(OnlineBase):
    def __init__(self, name, clf1, data, label, ftype, error):
        super(self.__class__, self).__init__(name, clf1, data, label, ftype, error)

    def do(self, n_pts):
        X, y = self.collect_pts(n_pts)

        print 'done collecting points'

        rbf_map = RBFSampler(n_components=n_pts, random_state=1)
        solver = HyperSolver(p=self.POS, n=self.NEG)
        rbf_solver = pipeline.Pipeline([("mapper", rbf_map),
                                        ("solver", solver)])

        gamma_range = np.logspace(-15, 6, 22, base=2)
        param_grid = dict(mapper__gamma=gamma_range)
        cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=1)
        grid = GridSearchCV(rbf_solver, param_grid=param_grid, cv=cv, n_jobs=8)
        grid.fit(X, y)

        scores = [x[1] for x in grid.grid_scores_]
        scores = np.array(scores).reshape(len(gamma_range))
        plt.figure(figsize=(8, 6))
        plt.plot(gamma_range, scores)

        plt.xlabel('gamma')
        plt.ylabel('score')
        plt.title('Validation accuracy (RTiX, %s)' % os.path.basename(self.name))
        plt.savefig(self.name + '-SLViF-grid-npts=%d.pdf' % n_pts)

        # final train
        g = grid.best_params_['mapper__gamma']
        print 'best parameters are g=%f' % g
        rbf_svc2 = grid.best_estimator_
        y_pred = rbf_svc2.predict(self.Xt)
        print 'SCORE: %f' % sm.accuracy_score(self.Yt, y_pred)
        return grid.best_score_, sm.accuracy_score(self.Yt, y_pred)


def run(train_data, test_data, n_features, gamma, C, feature_type='uniform'):
    X, Y = load_svmlight_file(train_data, n_features=n_features)
    Xt, Yt = load_svmlight_file(test_data, n_features=n_features)
    rbf_svc = svm.SVC(kernel='rbf', gamma=gamma, C=C).fit(X, Y)
    ex = GridRBFSolver(train_data, rbf_svc.predict, Xt, Yt, feature_type, 1e-9)
    ex.do(1500)


run('data/australian.aa', 'data/australian.ab', 14, gamma=0.03125, C=.125, feature_type='uniform')
