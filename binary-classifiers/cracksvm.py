__author__ = 'Fan'

from math import sqrt
import os
import time
import argparse
import sys
import logging

sys.path.append('libsvm-3.20/python')

from algorithms.RBFTrainer import RBFKernelRetraining
from algorithms.LinearTrainer import LinearTrainer
from algorithms.PolyTrainer import PolyTrainer
from algorithms.libsvmOnline import LibSVMOnline
from sklearn.datasets import load_svmlight_file
from utils.sanity import any_none

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run(name, n_features, train, validation, test, q):
    train_x, train_y = train
    val_x, val_y = validation
    test_x, test_y = test

    assert not any_none((train_x, train_y, val_x, val_y, test_x, test_y,))

    print q

    # start = time.time()

    # logger.info('Trying RBF kernel')
    rbf = RBFKernelRetraining(name, train_x, train_y, val_x, val_y, test_x, test_y, n_features)
    print rbf.grid_retrain_in_x()

    # poly = PolyTrainer(name, train_x, train_y, val_x, val_y, test_x, test_y, n_features)
    # print poly.grid_search()

    linear = LinearTrainer(name, train_x, train_y, val_x, val_y, test_x, test_y, n_features)
    print linear.grid_search()

    # now = time.time()
    # print 'time: %d %d (seconds)' % (len(train_y), now - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('-n', '--num_of_features', required=True, type=int)
    parser.add_argument('-s', '--step', type=int, default=6)
    args = vars(parser.parse_args())

    n_features = args['num_of_features']
    step = args['step']
    dataset_name = args['dataset']

    base_dir = os.path.join(os.getcwd(), 'targets/%s/' % dataset_name)
    model_file = os.path.join(base_dir, 'train.scale.model')

    # load model and collect QSV
    ex = LibSVMOnline(dataset_name, model_file, (1, -1), n_features, 'uniform', 1e-1)
    # generate test score
    test_x, test_y = load_svmlight_file(os.path.join(base_dir, 'test.scale'), n_features)
    test_x = test_x.todense()

    # ex.collect_pts(1000)
    #
    # run(dataset_name, n_features, (ex.pts_near_b, ex.pts_near_b_labels),
    #     (ex.support_pts, ex.support_labels), (test_x, test_y))

    train_x, train_y = [], []
    val_x, val_y = [], []
    while True:
        ex.collect_pts(step)
        train_x.extend(ex.pts_near_b)
        train_y.extend(ex.pts_near_b_labels)
        val_x.extend(ex.support_pts)
        val_y.extend(ex.support_labels)

        try:
            run(dataset_name, n_features, (train_x, train_y), (val_x, val_y), (test_x, test_y), ex.get_n_query())
        except KeyboardInterrupt:
            print 'Done'
            sys.exit()
