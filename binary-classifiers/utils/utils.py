from algorithms.RBFTrainer import RBFKernelRetraining
import os
from result import Result
from algorithms.libsvmOnline import LibSVMOnline
from sklearn.datasets import load_svmlight_file

def rbf_auto(ex, name, n_features, step):
    train_x, train_y = [], []
    val_x, val_y = [], []
    try:
        while True:
            ex.collect_pts(step)
            train_x.extend(ex.pts_near_b)
            train_y.extend(ex.pts_near_b_labels)
            val_x.extend(ex.support_pts)
            val_y.extend(ex.support_labels)
            e = RBFKernelRetraining(name, train_x, train_y, val_x, val_y, train_x, train_y, n_features)
            print ex.get_n_query(), e.grid_retrain_in_x()
    except KeyboardInterrupt:
        print 'Done'


