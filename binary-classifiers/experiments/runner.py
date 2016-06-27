import sys
sys.path.append('..')

from algorithms.RBFTrainer import RBFKernelRetraining
from baseline import Baseline
from utils.result import Result
from active_learning import ActiveLearning
import os
from algorithms.libsvmOnline import LibSVMOnline
from sklearn.datasets import load_svmlight_file


def libsvm_run(dataset_name, n_features, Extractor, ftype, n_repeat=5):
    n_features = float(n_features)
    base_dir = os.path.join(os.getcwd(), '../targets/%s/' % dataset_name)
    model_file = os.path.join(base_dir, 'train.scale.model')

    result = Result('%s-%s' %(dataset_name, Extractor.__name__))
    for repeat in range(0, n_repeat):
        print 'Round %d of %d'% (repeat, n_repeat - 1)

        # load model and collect QSV
        ex = LibSVMOnline(dataset_name, model_file, (1, -1), n_features, ftype, 1e-1)
        # generate test score
        X_test, y_test = load_svmlight_file(os.path.join(base_dir, 'test.scale'), n_features)
        X_test = X_test.todense()
        train_x, train_y = [], []
        for i in result.index:
            q_by_u = result.Q_by_U[i]

            ex.collect_up_to_budget(q_by_u * (n_features + 1))
            train_x.extend(ex.pts_near_b)
            train_y.extend(ex.pts_near_b_labels)
            main = Extractor(ex.batch_predict, (train_x, train_y), (X_test, y_test), n_features)

            L_unif, L_test = main.do()

            result.L_unif[i].append(L_unif)
            result.L_test[i].append(L_test)
            result.nquery[i].append(ex.get_n_query())

            # print ex.get_n_query() / (n_features + 1), ',', L_unif, ',', L_test

    print result

datasets = {
    'adult': (108, 'adult'),
    # 'australian': (14, 'australian'),
    # 'breast-cancer': (10, 'breast-cancer'),
    # 'circle': (2, 'circle'),
    # 'diabetes': (8, 'diabetes'),
    # 'fourclass': (2, 'fourclass'),
    # 'heart': (13, 'heart'),
    # 'moons': (2, 'moons'),
    # 'mushrooms': (112, 'mushrooms'),
}

import multiprocessing

if __name__ == '__main__':
    n_repeat = 1

    for k, v in datasets.items():
        # if k != 'fourclass':
        #     continue
        n_features, dataset_name = v
        # libsvm_run(dataset_name, n_features, Baseline, n_repeat)
        # d = multiprocessing.Process(target=libsvm_run, args=(dataset_name, n_features, Baseline, n_repeat))
        # d.start()
        # libsvm_run(dataset_name, float(n_features), RBFKernelRetraining, n_repeat)
        p = multiprocessing.Process(target=libsvm_run, args=(dataset_name, n_features, RBFKernelRetraining, 'binary', n_repeat))
        p.start()

