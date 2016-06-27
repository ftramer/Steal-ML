__author__ = 'Fan'
from math import sqrt

from sklearn.datasets import load_svmlight_file
from sklearn import svm
import sklearn.metrics as sm

from algorithms.OnlineBase import OnlineBase
from algorithms.PolySolver import PolySolver


def run(train_data, test_data, (p, n), n_features, f_type):
    X, Y = load_svmlight_file(train_data, n_features=n_features)
    X = X.todense()
    Xt, Yt = load_svmlight_file(test_data, n_features=n_features)
    Xt = Xt.todense()

    poly_svc = svm.SVC(kernel='poly', degree=2, gamma=1).fit(X, Y)

    baseline = sm.accuracy_score(Yt, poly_svc.predict(Xt))
    print 'BASELINE:    %f' % baseline

    def polynomial_map(x):
        # feature map for polynomial kernel (gamma* u`v + c)^2
        # assume gamma=1, c = 0
        n = len(x)
        r = []

        r.extend([x[i]*x[i] for i in range(n-1, -1, -1)])
        for i in range(n-1, -1, -1):
            for j in range(i-1, -1, -1):
                r.append(sqrt(2)*x[i]*x[j])
        return r

    print 'solve in F'
    online = OnlineBase(train_data, p, n, poly_svc.predict, n_features, f_type, 1e-5)
    online.collect_pts(-1, budget=5000)

    ex = PolySolver(online.get_QSV(), online.get_QSV_labels(), Xt, Yt, polynomial_map, n_features)
    ex.solve_in_f()
    print 'TRAIN SCORE  : %f' % ex.solve_score
    print 'TEST SCORE   : %f' % ex.calc_test_score()

    # print 'retrain in F'
    # ex = RBFKernelRetraining(train_data,
    #                   poly_svc.predict, Xt, Yt,
    #                          n_features, OfflineMethods.RT_in_F, error=1,
    #              kernel='poly', fmap=polynomial_map)
    # ex.train_SGD_for_poly_in_F()
    # ex.benchmark()
    # ex.print_perf()


# run('./data/mushrooms.aa', './data/mushrooms.ab', (1,2), 112, 'binary')
run('./data/australian.aa', './data/australian.ab', (-1, +1), 14, 'uniform')
