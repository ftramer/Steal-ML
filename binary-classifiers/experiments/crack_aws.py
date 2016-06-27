import sys
sys.path.append('..')
from algorithms.awsOnline import AWSOnline
from algorithms.RBFTrainer import RBFKernelRetraining
from sklearn.datasets import load_svmlight_file
from utils.result import Result

from baseline import Baseline
from active_learning import ActiveLearning

meta = {
    'breast-cancer': {
        'val_name': ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'],
        'model_id': 'ml-lkYRYeldcrH'
    },
    'circle' : {
        'val_name': ['x1', 'x2'],
        'model_id': 'ml-i0GeYZaGQ3f'
    },
    'fourclass' : {
        'val_name': ['x1', 'x2'],
        'model_id': 'ml-Je6DdX8c57P'
    },
    'diabetes': {
        'val_name': ['x' + str(i) for i in range(1, 10 + 1)],
        'model_id': 'ml-UGnrMStrX2o'
    }
}


n_repeat = 1


def run(dataset):
    n_features = len(meta[dataset]['val_name'])

    result_online = Result('%s-%s' %(dataset, 'aws-online'), aws=True)
    result_baseline = Result('%s-%s' %(dataset, 'aws-baseline'), aws=True)
    result_active = Result('%s-%s' %(dataset, 'aws-active'), aws=True)

    for repeat in range(0, n_repeat):
        print 'Round %d of %d'% (repeat, n_repeat - 1)

        ex = AWSOnline(meta[dataset]['model_id'], 1, 0, n_features, meta[dataset]['val_name'], ftype='uniform', error=.1)

        test_x, test_y = load_svmlight_file('/Users/Fan/dev/ML/code/binary-classifiers/targets/%s/test.scale' % dataset, n_features)
        test_x = test_x.todense()
        test_y = [a if a == 1 else 0 for a in test_y]
        train_x, train_y = [], []

        for i in result_active.index:
            q_by_u = result_active.Q_by_U[i]
            print 'Active learning with budget %d / %d' % (q_by_u, q_by_u * (n_features + 1))
            main = ActiveLearning(ex, (None, None), (test_x, test_y), n_features,
                                  q_by_u * (n_features + 1), 5)

            L_unif, L_test = main.do()

            result_active.L_unif[i].append(L_unif)
            result_active.L_test[i].append(L_test)
            result_active.nquery[i].append(ex.get_n_query())

        ex = AWSOnline(meta[dataset]['model_id'], 1, 0, n_features, meta[dataset]['val_name'], ftype='uniform', error=.1)

        for i in result_online.index:
            q_by_u = result_online.Q_by_U[i]
            print 'collecting up to budget %d / %d' % (q_by_u, q_by_u * (n_features + 1))

            ex.collect_up_to_budget(q_by_u * (n_features + 1))
            train_x.extend(ex.pts_near_b)
            train_y.extend(ex.pts_near_b_labels)

            print 'retraining with %d points' % len(train_y)

            # online
            e = RBFKernelRetraining(ex.batch_predict, (train_x, train_y), (test_x, test_y), n_features)
            L_unif, L_test = e.grid_retrain_in_x()

            result_online.L_unif[i].append(L_unif)
            result_online.L_test[i].append(L_test)
            result_online.nquery[i].append(ex.get_n_query())

            # baseline
            e = Baseline(ex.batch_predict, (train_x, train_y), (test_x, test_y), n_features)
            L_unif, L_test = e.do()

            result_baseline.L_unif[i].append(L_unif)
            result_baseline.L_test[i].append(L_test)
            result_baseline.nquery[i].append(ex.get_n_query())

    print result_online
    print result_baseline
    print result_active

for k, v in meta.items():
    if k != 'breast-cancer':
        continue
    run(k)