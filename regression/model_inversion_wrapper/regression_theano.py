import argparse
from sklearn.metrics import accuracy_score
from theano_softmax import SoftmaxExtractor
import theano_softmax
import numpy as np
from collections import Counter
import utils
import sys
from pylearn2.utils import serial
from regression_stealer import * 



class TheanoMLInversionExtractor(SoftmaxExtractor):

    def __init__(self, model_file):
        self.model = serial.load(model_file)
        self.multinomial = True
        self.w, self.intercept = self.get_coefficients()
        self.use_intercept = True
        SoftmaxExtractor.__init__(self, 'faces')

    def num_features(self):
        return self.w.shape[1]

    def get_classes(self):
        return np.array(range(len(self.intercept)))

    def get_coefficients(self):
        w = self.model.layers[0].get_weights().T
        intercept = self.model.layers[0].get_biases()
        return w, intercept

    def query_probas(self, X):
        p = predict_probas(X, self.w, self.intercept,
                           multinomial=self.multinomial)
        return p

    def query(self, X):
        return predict_classes(X, self.w, self.intercept, self.get_classes())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str, help='a pickled model file')
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

    model_file = args.model_file
    action = args.action
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

    X_train, y_train, X_test, y_test, _ = utils.prepare_data('att_faces')

    ext = TheanoMLInversionExtractor(model_file)

    num_unknowns = len(ext.get_classes()) * ext.num_features()

    try:
        budget = int(budget)
    except ValueError:
        budget = int(float(budget) * num_unknowns)

    try:
        steps = map(int, steps)
    except ValueError:
        steps = map(lambda x: int(float(x) * num_unknowns), steps)

    print >> sys.stderr, 'Data: {}, Action: {}, Budget:{}, Seed: {}'.\
        format(model_file, action, budget, seed)
    print >> sys.stderr, 'Number of unknowns: {}'.format(num_unknowns)

    if action == "extract":
        ext.extract(budget, steps=steps, print_epoch=1, 
                    adaptive_oracle=adaptive_oracle, num_passes=num_passes,
                    epsilon=epsilon, batch_size=batch_size, random_seed=seed)
    elif action == "compare":
        X_test_u = utils.gen_query_set(X_test.shape[1], 1000)
        ext.compare(X_test, X_test_u)
    else:
        raise ValueError('Unknown action')

    
if __name__ == "__main__":
    main()
