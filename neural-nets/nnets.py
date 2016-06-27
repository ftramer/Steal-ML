import argparse
from sklearn.metrics import accuracy_score
import theano_mlp as perceptron
from theano_mlp import PerceptronExtractor
import numpy as np
from collections import Counter
import utils
try:
    import matplotlib.pyplot as plt
except:
    plt = None
    pass
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder


"""
Moons
=====
Train:      eps=1e-2, 1000 passes, 94% accuracy
Extract:    eps=1e-2, 2000 passes, 60 samples (100% - 94% baseline)


Blobs
=====
Train:      eps=1e-2, 100 passes, 100% accuracy
Extract:    eps=1e-2, 2000 passes, 50 samples (100% - 91% baseline)


Circles
=======
Train:      eps=1e-2, 1000 passes, 100% accuracy
Extract:    eps=1e-2, 2000 passes, 60 samples (100% - 90% baseline)


Faces
=====
* 20 hidden nodes

Train:      eps=1e-2, 1000 passes, 94% accuracy
Extract:    eps=1e-2, 2000 passes, 10000 samples (97.5% - 60% baseline)
Extract:    eps=1e-1, 1000 passes, 20000 samples (100% - 66% baseline)


MNIST
=====

* 100 hidden nodes

Train:      eps=1e-2, 100 passes, 97% accuracy
Extract:    eps=1e-2, 1000 passes, 20000 samples (???)

"""


class LocalPerceptronExtractor(PerceptronExtractor):

    def __init__(self, dataset, hidden_nodes, X_train, y_train, rounding,
                 force_reg=False):
        self.X_train = X_train
        self.y_train = y_train
        self.classes = pd.Series(y_train).unique()
        self.model = None
        self.rounding = rounding
        self.force_reg = force_reg

        PerceptronExtractor.__init__(self, dataset, hidden_nodes)

    def load_model(self):
        self.model = perceptron.load('experiments/{}/models/oracle_{}.pkl'.
                                     format(self.dataset, self.hidden_nodes),
                                     force_reg=self.force_reg)

    def num_features(self):
        return self.X_train.shape[1]

    def get_classes(self):
        return self.classes

    def query_probas(self, X):
        if self.model is None:
            self.load_model()
        p = perceptron.predict_probas(self.model, X)
        if self.rounding is None:
            return p
        else:
            return np.round(p, self.rounding)

    def query(self, X):
        if self.model is None:
            self.load_model()
        return perceptron.predict(self.model, X)

    def calculate_loss(self, X, y, reg):
        if self.model is None:
            self.load_model()
        return perceptron.calculate_loss(self.model, X, y, reg)

    def train(self, X_test, y_test, num_passes=100):
        X_train = self.X_train
        num_classes = len(self.get_classes())

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(self.y_train)
        y_test = encoder.transform(y_test)

        y_p = np.zeros((len(y_train), num_classes))
        y_p[np.arange(len(y_train)), y_train] = 1

        model_bb = perceptron.build_model(self.hidden_nodes, X_train, y_p,
                                          num_passes=num_passes, epsilon=1e-2,
                                          epoch=100,
                                          eps_factor=1.0,
                                          print_loss=True, print_epoch=10,
                                          batch_size=20,
                                          force_reg=self.force_reg)

        perceptron.save(model_bb, 'experiments/{}/models/oracle_{}.pkl'.
                        format(self.dataset, self.hidden_nodes))

        y_pred = perceptron.predict(model_bb, X_test)
        print Counter(y_pred)
        acc = accuracy_score(y_test, y_pred)
        print 'Training accuracy: {}'.format(acc)

        if X_train.shape[1] == 2 and plt is not None:
            bounds = [-1.1, 1.1, -1.1, 1.1]
            X_train = X_train.values
            plt.figure()
            plt.scatter(X_train[:, 0], X_train[:, 1], s=40, c=y_train,
                        cmap=plt.cm.Spectral)
            plt.savefig('experiments/{}/plots/data_{}'.
                        format(self.dataset, len(X_train)))

            filename = 'experiments/{}/plots/oracle_{}_boundary'.\
                format(self.dataset, self.hidden_nodes)
            utils.plot_decision_boundary(lambda x:
                                         perceptron.predict(model_bb, x),
                                         X_train, y_train, bounds, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='a dataset')
    parser.add_argument('hidden_nodes', type=int, help='number of hidden nodes')
    parser.add_argument('action', type=str, help='action to perform')
    parser.add_argument('budget', type=str, help='query budget')
    parser.add_argument('--num_passes', type=int, help='number of passes',
                        default=1000)
    parser.add_argument('--epsilon', type=float, help='learning rate',
                        default=0.1)
    parser.add_argument('--rounding', type=int, help='rounding digits')
    parser.add_argument('--steps', type=str, nargs='+', default=[],
                        help='adaptive active learning')
    parser.add_argument('--adaptive_oracle', dest='adaptive_oracle',
                        action='store_true',
                        help='adaptive active learning from oracle')
    parser.add_argument('--force_reg', dest='force_reg',
                        action='store_true',
                        help='train a regression layer only')
    parser.add_argument('--batch_size', type=int, help='batch size', default=1)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    dataset = args.data
    action = args.action
    hidden_nodes = args.hidden_nodes
    budget = args.budget
    num_passes = args.num_passes
    rounding = args.rounding
    steps = args.steps
    adaptive_oracle = args.adaptive_oracle
    epsilon = args.epsilon
    batch_size = args.batch_size
    force_reg = args.force_reg
    seed = args.seed

    np.random.seed(0)

    X_train, y_train, X_test, y_test, scaler = utils.prepare_data(dataset)

    if force_reg:
        dataset += "_reg"

    ext = LocalPerceptronExtractor(dataset, hidden_nodes, X_train, y_train,
                                   rounding=rounding, force_reg=force_reg)

    num_unknowns = hidden_nodes * (X_train.shape[1] + 1) + \
                   len(ext.get_classes()) * (hidden_nodes + 1)

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
        ext.train(X_test, y_test, num_passes=num_passes)
    elif action == "extract":
        ext.extract(X_train, y_train, budget, steps=steps,
                    adaptive_oracle=adaptive_oracle, num_passes=num_passes,
                    epsilon=epsilon, batch_size=batch_size, random_seed=seed)
    elif action == "baseline":
        ext.extract(X_train, y_train, budget, steps=steps,
                    adaptive_oracle=adaptive_oracle, baseline=True,
                    num_passes=num_passes, epsilon=epsilon,
                    batch_size=batch_size, random_seed=seed,
                    reg_lambda=1e-40)
    elif action == "compare":
        X_test_u = utils.gen_query_set(X_test.shape[1], 10000)
        ext.compare(X_test, X_test_u, force_reg=force_reg)
    else:
        raise ValueError('Unknown action')

    
if __name__ == "__main__":
    main()
