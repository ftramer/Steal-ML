__author__ = 'Fan'

import pickle
from math import sqrt
import os
import glob
import time

from sklearn.datasets import load_svmlight_file

from algorithms.RBFTrainer import RBFKernelRetraining
from algorithms.LinearTrainer import LinearTrainer
from algorithms.OfflineBase import OfflineMethods
from algorithms.PolySolver import PolySolver


def run(n_feature, train, validation, test):
    train_x, train_y = train
    val_x, val_y = validation
    test_x, test_y = test


    start = time.time()
    ex = RBFKernelRetraining('aws', train_x, train_y,
                             val_x, val_y,
                             test_x, test_y, n_feature, OfflineMethods.RT_in_F)
    print ex.grid_retrain_in_x()
    # print ex.grid_retrain_in_f(100)

    def quadratic_map(x):
        # feature map for polynomial kernel (gamma* u`v + c)^2
        # assume gamma=1, c = 0
        n = len(x)
        r = []
        r.extend([x[i] * x[i] for i in range(n - 1, -1, -1)])
        for i in range(n - 1, -1, -1):
            for j in range(i - 1, -1, -1):
                r.append(sqrt(2) * x[i] * x[j])
        return r

    # TODO retrain directly and plot into plots
    poly = PolySolver('aws', train_x, train_y, val_x, val_y, test_x, test_y, quadratic_map, n_feature)
    print poly.grid_retrain_in_f()

    linear = LinearTrainer('aws', train_x, train_y, val_x, val_y, test_x, test_y, n_feature)
    print linear.grid_search()

    now = time.time()
    print 'time: %d %d' % (len(train_y), now - start)


n_feature = 10
base_dir = os.getenv('HOME') + '/Dropbox/Projects/SVM/ml-lkYRYeldcrH/'
q = glob.glob(base_dir + 'e0.1b50s*')

from random import shuffle
shuffle(q)

test_x, test_y = load_svmlight_file(os.getenv('HOME') + '/Dropbox/Projects/SVM/dataset/breast-cancer/bc-test',
                                    n_features=n_feature)
test_x = test_x.todense()

val_x = test_x[:50]
val_y = test_y[:50]
for r in range(0, len(q)):
    b = []  # boundary points
    bl = []
    s = []  # support points
    sl = []
    fs = q[:r + 1]
    for f in fs:
        with open(f) as inf:
            _b, _bl, _s, _sl = pickle.load(inf)
            b.extend(_b)
            bl.extend(_bl)
            s.extend(_s)
            sl.extend(_sl)

    run(n_feature, (s, sl), (val_x, val_y), (test_x, test_y))
