__author__ = 'Fan'

from math import sqrt
import os
import time
import argparse
import sys
import logging

from algorithms.RBFTrainer import RBFKernelRetraining
from algorithms.LinearTrainer import LinearTrainer
from algorithms.PolyTrainer import PolyTrainer
from algorithms.libsvmOnline import LibSVMOnline
from sklearn.datasets import load_svmlight_file
from utils.sanity import any_none

from utils.result import Result

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    n_features = 10
    dataset_name = 'breast-cancer'

    base_dir = os.path.join(os.getcwd(), '../targets/%s/' % dataset_name)
    model_file = os.path.join(base_dir, 'train.scale.model')

    # load model and collect QSV
    ex = LibSVMOnline(dataset_name, model_file, (1, -1), n_features, 'uniform', 1e-1)
    # generate test score
    X_test, y_test = load_svmlight_file(os.path.join(base_dir, 'test.scale'), n_features)
    X_test = X_test.todense()

    train_x, train_y = [], []

    for i in range(0, 10):
        ex.collect_up_to_budget(50)
        train_x.extend(ex.pts_near_b)
        train_y.extend(ex.pts_near_b_labels)

        rbf = RBFKernelRetraining(dataset_name, ex.batch_predict, train_x, train_y, X_test, y_test, n_features)
        print 'Q = ', ex.get_n_query()
        rbf.grid_retrain_in_x()


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