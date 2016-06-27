import pandas as pd
from kernel_regression_stealer import KernelRegressionExtractor
import kernel_regression_stealer as krs
import numpy as np
import theano
import theano.tensor as T
import utils
from sklearn.metrics import accuracy_score
import argparse
from collections import Counter
import sys


class LocalKernelExtractor(KernelRegressionExtractor):

    def __init__(self, dataset, X_train, y_train, rounding):
        self.X_train = X_train
        self.y_train = y_train
        self.classes = pd.Series(y_train).unique()
        self.model = None
        self.rounding = rounding

        KernelRegressionExtractor.__init__(self, dataset)

    def load_model(self):
        self.model = krs.load('experiments/KLR/{}/models/oracle.pkl'.
                              format(self.dataset))

    def get_gamma(self):
        if self.model is None:
            self.load_model()
        return self.model.kernelLayer.gamma

    def num_features(self):
        return self.X_train.shape[1]

    def get_classes(self):
        return self.classes

    def query_probas(self, X):
        if self.model is None:
            self.load_model()
        p = krs.predict_probas(self.model, X)
        if self.rounding is None:
            return p
        else:
            return np.round(p, self.rounding)

    def query(self, X):
        if self.model is None:
            self.load_model()
        return krs.predict(self.model, X)

    def calculate_loss(self, X, y, reg):
        if self.model is None:
            self.load_model()
        return krs.calculate_loss(self.model, X, y, reg)

    def train(self, num_repr, X_test, y_test):
        X_train = self.X_train
        y_train = self.y_train
        y_train_p = np.zeros((len(y_train), len(self.classes)))
        y_train_p[np.arange(len(y_train)), y_train] = 1

        """
        assert num_repr >= len(self.get_classes())

        X_repr_bb = []
        class_counter = Counter(y_train)

        repr_per_class \
            = (num_repr + len(self.get_classes()) - 1) / len(self.get_classes())

        for (c, count) in sorted(class_counter.iteritems(), key=lambda _: _[1]):
            print c, count, repr_per_class
            reprs = X_train.values[np.where(y_train == c)[0][0:repr_per_class]]
            X_repr_bb.append(reprs)

        X_repr_bb = np.vstack(X_repr_bb)[0:num_repr]
        print '{} representers'.format(len(X_repr_bb))
        """

        X_repr_bb = X_train[0:num_repr].values

        """
        import math
        import matplotlib.pyplot as plt
        side = math.sqrt(X_repr_bb.shape[1])
        plt.figure()
        for i in range(len(X_repr_bb)):
            plt.subplot(2, len(X_repr_bb)/2, i + 1)
            plt.axis('off')
            image = X_repr_bb[i, :].reshape((side, side))
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.show()
        """
        """
        model_bb = krs.build_model(X_repr_bb, False, X_train, y_train_p,
                                   epsilon=epsilon, reg_lambda=1e-4,
                                   num_passes=num_passes, eps_factor=0.99,
                                   epoch=100, print_epoch=10, batch_size=1,
                                   gamma=gamma)

        y_pred_bb = krs.predict(model_bb, X_test)
        """

        best_model = None
        best_acc = 0
        best_gamma = None

        for gamma in [10**x for x in range(-10, 10)]:
            from sklearn.metrics.pairwise import rbf_kernel
            from sklearn.linear_model import LogisticRegression
            X_train_ker = rbf_kernel(X_train, X_repr_bb, gamma=gamma)
            model = LogisticRegression(multi_class='multinomial',
                                       solver='lbfgs').\
                fit(X_train_ker, y_train)

            y_pred = model.predict(rbf_kernel(X_test, X_repr_bb, gamma=gamma))
            acc = accuracy_score(y_test, y_pred)

            if acc > best_acc:
                best_acc = acc
                best_gamma = gamma
                best_model = model

        W = best_model.coef_.T
        b = best_model.intercept_

        if len(self.classes) == 2:
            W = np.hstack((np.zeros((len(X_repr_bb), 1)), W))
            b = np.hstack((0, b))
        W = theano.shared(
                value=W,
                name='W',
                borrow=True
            )
        b = theano.shared(
                value=b,
                name='b',
                borrow=True
            )

        model_bb = krs.KernelLog(T.matrix('x'), best_gamma, len(self.classes),
                                 X_repr_bb, learn_Y=False, W=W, b=b)
        y_pred_bb = krs.predict(model_bb, X_test)

        print 'Best Gamma: {}'.format(best_gamma)
        print 'Y_test: {}'.format(Counter(y_test))
        print 'Y_pred: {}'.format(Counter(y_pred_bb))

        acc = accuracy_score(y_test, y_pred_bb)
        print 'Training accuracy: {}'.format(acc)
        print >> sys.stderr, 'Training accuracy: {}'.format(acc)

        X_test_u = utils.gen_query_set(X_test.shape[1], 10000)
        y_pred_u = krs.predict(model_bb, X_test_u)

        print 'Y_pred_u: {}'.format(Counter(y_pred_u))

        if X_train.shape[1] == 2:
            bounds = [-1.1, 1.1, -1.1, 1.1]
            X_train = X_train.values
            utils.plot_decision_boundary(
                lambda x: krs.predict(model_bb, x), X_train, y_train, bounds)

        krs.save(model_bb,
                 'experiments/KLR/{}/models/oracle.pkl'.format(self.dataset))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='a dataset')
    parser.add_argument('action', type=str, help='action to perform')
    parser.add_argument('num_repr', type=int, help='number of representers')
    parser.add_argument('budget', type=str, help='query budget')
    parser.add_argument('--num_passes', type=int, help='number of passes',
                        default=1000)
    parser.add_argument('--rounding', type=int, help='rounding digits')
    parser.add_argument('--steps', type=str, nargs='+', default=[],
                        help='adaptive active learning')
    parser.add_argument('--adaptive_oracle', dest='adaptive_oracle',
                        action='store_true',
                        help='adaptive active learning from oracle')
    parser.add_argument('--gamma', type=float,
                        help='RBF kernel hyper-parameter')
    parser.add_argument('--epsilon', type=float, help='learning rate',
                        default=0.1)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, help='batch size')
    args = parser.parse_args()

    dataset = args.data
    action = args.action
    num_repr = args.num_repr
    budget = args.budget
    num_passes = args.num_passes
    rounding = args.rounding
    steps = args.steps
    adaptive_oracle = args.adaptive_oracle
    gamma = args.gamma
    epsilon = args.epsilon
    seed = args.seed
    batch_size = args.batch_size

    np.random.seed(0)

    X_train, y_train, X_test, y_test, scaler = utils.prepare_data(dataset)
    X_test_u = utils.gen_query_set(X_test.shape[1], 1000)
    ext = LocalKernelExtractor(dataset, X_train, y_train, rounding=rounding)

    num_unknowns = num_repr * X_train.shape[1] + \
                   len(ext.get_classes()) * (num_repr + 1)
    try:
        budget = int(budget)
    except ValueError:
        budget = int(float(budget) * num_unknowns)

    try:
        steps = map(int, steps)
    except ValueError:
        steps = map(lambda x: int(float(x) * num_unknowns), steps)

    print >> sys.stderr, 'Data: {}, Action: {}, Budget:{}, Seed: {}'.\
        format(dataset, action, budget, seed)
    print >> sys.stderr, 'Number of unknowns: {}'.format(num_unknowns)

    if action == "train":
        ext.train(num_repr, X_test, y_test)
    elif action == "extract":
        if gamma is None:
            gamma = ext.get_gamma()

        print gamma
        ext.extract(X_train, y_train, num_repr, budget, gamma=gamma, steps=steps,
                    adaptive_oracle=adaptive_oracle, num_passes=num_passes,
                    epsilon=epsilon, random_seed=seed, batch_size=batch_size)
    elif action == "baseline":
        if gamma is None:
            gamma = ext.get_gamma()
        ext.extract(X_train, y_train, num_repr, budget, gamma=gamma, steps=steps,
                    adaptive_oracle=adaptive_oracle, baseline=True,
                    num_passes=num_passes, epsilon=epsilon, random_seed=seed,
                    batch_size=batch_size, reg_lambda=1e-40)
    elif action == "compare":
        ext.compare(X_test, X_test_u, scaler=None)


if __name__ == "__main__":
    main()
