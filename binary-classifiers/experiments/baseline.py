from algorithms.libsvmOnline import LibSVMOnline

__author__ = 'Fan'

import logging
import os
import sys
import numpy as np

from sklearn import svm
from sklearn.datasets import load_svmlight_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from algorithms.OnlineBase import OnlineBase
from algorithms.OfflineBase import OfflineBase

from utils.result import Result


class Baseline(OfflineBase):
    def __init__(self, oracle, retrain_xy, test_xy, n_features):
        X_ex, y_ex = retrain_xy
        X_test, y_test = test_xy

        super(self.__class__, self).__init__(
            oracle, X_ex, y_ex, X_test, y_test, n_features
        )

        if 0 in self.y_test:
            self.NEG = 0
        elif -1 in self.y_test:
            self.NEG = -1
        else:
            print 'Watch out for test file! Neither 0 nor 1 is included!'

    def do(self):
        if len(np.unique(self.y_ex)) < 2:
            return 1,1

        clf2 = svm.SVC(C=1e5)
        clf2.fit(self.X_ex, self.y_ex)

        self.set_clf2(clf2)

        return self.benchmark()


def run(dataset_name, n_features):
    base_dir = os.path.join(os.getcwd(), '../targets/%s/' % dataset_name)
    model_file = os.path.join(base_dir, 'train.scale.model')

    result = Result('baseline')
    n_repeat = 10
    for repeat in range(0, n_repeat):
        print 'Round %d of %d'% (repeat, n_repeat - 1)

        # load model and collect QSV
        ex = LibSVMOnline(dataset_name, model_file, (1, -1), n_features, 'uniform', 1e-1)
        # generate test score
        X_test, y_test = load_svmlight_file(os.path.join(base_dir, 'test.scale'), n_features)
        X_test = X_test.todense()
        train_x, train_y = [], []
        for i in result.index:
            q_by_u = result.Q_by_U[i]
            ex.collect_up_to_budget(q_by_u * (n_features + 1))
            train_x.extend(ex.pts_near_b)
            train_y.extend(ex.pts_near_b_labels)
            base = Baseline(ex.batch_predict, (train_x, train_y), (X_test, y_test), n_features)

            L_unif, L_test = base.do()

            result.L_unif[i].append(L_unif)
            result.L_test[i].append(L_test)
            result.nquery[i].append(ex.get_n_query())

            # print ex.get_n_query() / (n_features + 1), ',', L_unif, ',', L_test

    print result