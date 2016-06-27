__author__ = 'Fan'

import logging
import os
from sklearn import svm
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

from algorithms.OnlineBase import OnlineBase
from algorithms.libsvmOnline import LibSVMOnline

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from utils.result import Result

from algorithms.OfflineBase import OfflineBase


class ActiveLearning (OfflineBase):
    def __init__(self, ex, retrain_xy, test_xy, n_features, total_budget, n_rounds):
        self.total_budget = total_budget
        self.n_rounds = n_rounds
        self.budget_per_round = int(self.total_budget / self.n_rounds)

        self.ex = ex

        X_ex, y_ex = retrain_xy
        X_test, y_test = test_xy

        super(self.__class__, self).__init__(
            ex.batch_predict, X_ex, y_ex, X_test, y_test, n_features
        )

        if 0 in self.y_test:
            self.NEG = 0
        elif -1 in self.y_test:
            self.NEG = -1
        else:
            print 'Watch out for test file! Neither 0 nor 1 is included!'

    def do(self):
        # get some initial points
        self.ex.collect_up_to_budget(self.budget_per_round * 2)
        x, y = self.ex.pts_near_b, self.ex.pts_near_b_labels

        if len(np.unique(y)) < 2:
            return 1, 1

        # gamma_range = np.logspace(-5, 1, 10, base=10)
        # param_grid = dict(gamma=gamma_range)

        try:
            # cv = StratifiedShuffleSplit(y, n_iter=5, test_size=.2)
            # grid = GridSearchCV(svm.SVC(C=1e5), param_grid=param_grid, cv=cv, n_jobs=-1)
            # grid.fit(x, y)
            # h_best = grid.best_estimator_
            raise ValueError
        except ValueError:
            h_best = svm.SVC(C=1e5)
            h_best.fit(x, y)

        for i in range(1, self.n_rounds - 1):
            online_ = OnlineBase('', +1, self.NEG, h_best.predict, self.n_features, 'uniform', error=.1)
            x_, _ = online_.collect_pts(self.budget_per_round, 50000)  # budget doesn't matter

            xx_ = None
            if x_ is None or len(x_) < self.budget_per_round:
                print('Run out of budget when getting x_')
                xx_ = np.random.uniform(-1, 1, (self.budget_per_round - len(x_), self.n_features))

            if x_ is not None and len(x_) > 0:
                x.extend(x_)
                y.extend(self.oracle(x_))

            if xx_ is not None:
                x.extend(xx_)
                y.extend(self.oracle(xx_))

            try:
                # cv = StratifiedShuffleSplit(y, n_iter=5, test_size=.2)
                # grid = GridSearchCV(svm.SVC(C=1e5), param_grid=param_grid, cv=cv, n_jobs=-1)
                # grid.fit(x, y)
                # h_best = grid.best_estimator_
                raise ValueError
            except ValueError:
                h_best = svm.SVC(C=1e5)
                h_best.fit(x, y)

            # h_best.fit(x, y)

        self.set_clf2(h_best)
        return self.benchmark() # (ex.batch_predict, h_.predict, test_x, n_features)


def run(dataset_name, n_features, n_repeat=5, n_learning_round=5):
    base_dir = os.path.join(os.getcwd(), '../targets/%s/' % dataset_name)
    model_file = os.path.join(base_dir, 'train.scale.model')

    result = Result(dataset_name + '-'+ 'active')
    for repeat in range(0, n_repeat):
        print 'Round %d of %d'% (repeat, n_repeat - 1)

        ex = LibSVMOnline(dataset_name, model_file, (1, -1), n_features, 'uniform', 1e-1)
        X_test, y_test = load_svmlight_file(os.path.join(base_dir, 'test.scale'), n_features)
        X_test = X_test.todense()

        for i in result.index:
            q_by_u = result.Q_by_U[i]

            main = ActiveLearning(ex, (None, None), (X_test, y_test), n_features,
                                  q_by_u * (n_features + 1), n_learning_round)

            L_unif, L_test = main.do()

            result.L_unif[i].append(L_unif)
            result.L_test[i].append(L_test)
            result.nquery[i].append(ex.get_n_query())

    print result


datasets = {
    'adult': (123, 'adult'),
    'australian': (14, 'australian'),
    'breast-cancer': (10, 'breast-cancer'),
    'circle': (2, 'circle'),
    'diabetes': (8, 'diabetes'),
    'fourclass': (2, 'fourclass'),
    'heart': (13, 'heart'),
    'moons': (2, 'moons'),
    'mushrooms': (112, 'mushrooms'),
}

import multiprocessing

if __name__ == '__main__':
    for k, v in datasets.items():
        n_features, dataset_name = v
        p = multiprocessing.Process(target=run, args=(dataset_name, n_features,))
        p.start()