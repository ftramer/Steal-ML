from models import models
import boto3
from regression_stealer import RegressionExtractor
import numpy as np
import argparse
import utils
import cPickle
from copy import copy
import decimal
import re
import timeit
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import logging
import sys


class Bin(object):
    def __init__(self, lower, upper, eps):
        self.lower = lower
        self.upper = upper
        self.eps = eps
        self.repr = []

    def contains(self, val):
        if self.lower == self.upper == val:
            return True
        else:
            return self.lower < val <= self.upper

    def repr_val(self, size=1):
        """
        if self.upper-self.lower < 2*self.eps:
            return np.array([self.upper] * size)

        return [0.5 * (self.lower + self.upper)] * size
        """
        return np.array([self.upper] * size)

    def unif_val(self, size=1):

        if self.upper-self.lower < 2*self.eps:
            return np.array([self.upper] * size)

        #return [0.5 * (self.lower + self.upper)] * size
        return np.random.uniform(self.lower, self.upper, size)

    def __repr__(self):
        return 'Bin({}, {})'.format(self.lower, self.upper)


def make_query(query, feature_name, val):
    """
    Modify a given query by switching the value of one feature
    """
    new_query = copy(query)
    new_query[feature_name] = val
    return new_query


def eps_round(x, epsilon):
    """
    Round a floating point value to the nearest multiple of eps
    """
    return round(x / epsilon) * epsilon


class AWSRegressionExtractor(RegressionExtractor):

    def __init__(self, model_id, X, cat_idx, incomplete=True, eps=1e-3):
        self.model_id = model_id
        self.client = boto3.client('machinelearning')

        self.model = self.client.get_ml_model(
            MLModelId=models[model_id]['id'],
            Verbose=True
        )

        schema = self.model['Schema']

        target_reg = '"targetAttributeName":"(.*?)"'
        target = re.findall(target_reg, schema)[0]

        feature_reg = '"attributeName":"(.*?)","attributeType":"(.*?)"'
        feature_info = re.findall(feature_reg, schema)

        features = [(f, t) for (f, t) in feature_info if f != target]

        cat_features = [(f, t) for (f, t) in features if t == "CATEGORICAL"]
        numeric_features = [(f, t) for (f, t) in features if t == "NUMERIC"]

        self.input_features = [f for (f, t) in cat_features + numeric_features]
        self.feature_types = [t for (f, t) in cat_features + numeric_features]

        self.one_hot = len(cat_features) > 0
        logging.info('ONE HOT ENCODING: {}'.format(self.one_hot))

        self.label_encoders = {}

        for i in range(len(cat_idx)):
            self.label_encoders[i] = LabelEncoder()
            X[:, i] = self.label_encoders[i].fit_transform(X[:, i])

        self.encoder = OneHotEncoder(categorical_features=range(len(cat_idx)),
                                     sparse=False)
        self.encoder.fit(X)

        self.classes = np.array(models[model_id]['classes'])
        self.multinomial = len(self.classes) != 2

        self.eps = eps
        self.query_count = 0

        self.query_cache = {}

        """
        self.cache_file = 'aws_wrapper/cache/{}.pkl'.format(model_id)
        try:
            with open(self.cache_file, 'rb') as f:
                self.query_cache = cPickle.load(f)
        except IOError:
            logging.info('Creating Query Cache')
        """

        self.binning = '_QB_' in self.model['Recipe']
        logging.info('BINNING: {}'.format(self.binning))
        if self.binning:
            self.bins = self.find_bins()
        else:
            self.bins = {}

        logging.info('Input features: {}: {}'.format(self.num_input_features(),
                                                     self.input_features))
        logging.info('Feature Space: {}'.format(self.num_features()))

        self.incomplete = incomplete

        RegressionExtractor.__init__(self)

    def num_features(self):
        if not (self.binning or self.one_hot):
            return len(self.input_features)
        else:
            tot = sum([len(b) for b in self.bins.values()])
            if self.one_hot:
                tot += sum(self.encoder.n_values_)
            return tot

    def num_input_features(self):
        return len(self.input_features)

    def get_classes(self):
        return self.classes

    def make_ls_pred(self, q):
        return self.aws_query(np.array([q]))[0]['predictedScores'].values()

    def line_search(self, query, feature, min_val, max_val, min_p, max_p):
        """
        Perform a line search on a continuous feature to find all splitting
        thresholds
        """
        logging.info('\tmin val {} got {}'.format(min_val, min_p))
        logging.info('\tmax val {} got {}'.format(max_val, max_p))
        prec = -decimal.Decimal(str(self.eps)).as_tuple().exponent

        def int_line_search(query, feature, min_mult, max_mult, min_p, max_p):
            if min_p == max_p:
                split = (min_mult + max_mult)/2
                split_query = make_query(query, feature,
                                         round(split * self.eps, prec))
                split_p = self.make_ls_pred(split_query)

                if split_p == min_p:
                    return []

            # if we've narrowed down the split to less than epsilon,
            # call it a day
            if (max_mult - min_mult) <= 1:
                threshold = round(min_mult * self.eps, prec)

                logging.info('ending line search with {} ({} - {})'.format(
                    threshold, min_mult, max_mult))

                return [threshold]

            # split at half way and check recursively on both sides
            split = (min_mult + max_mult)/2
            split_query = make_query(query, feature,
                                     round(split * self.eps, prec))
            split_p = self.make_ls_pred(split_query)

            logging.info('\tval {} got {}'.format(round(split*self.eps, prec),
                                                  split_p))

            return int_line_search(query, feature, split,
                                   max_mult, split_p, max_p) + \
                   int_line_search(query, feature, min_mult,
                                   split, min_p, split_p)

        max_mult = int(round(max_val / self.eps))
        min_mult = int(round(min_val / self.eps))
        return int_line_search(query, feature, min_mult, max_mult, min_p, max_p)

    def find_bins(self):
        start_time = timeit.default_timer()
        thresholds = {}

        q0 = [None] * len(self.input_features)
        for i, f in enumerate(self.input_features):
            if self.feature_types[i] != 'NUMERIC':
                continue

            logging.info('line search on feature {}'.format(f))
            val_min = -1.1
            q_min = make_query(q0, i, val_min)
            p_min = self.make_ls_pred(q_min)

            val_max = 1.1
            q_max = make_query(q0, i, val_max)
            p_max = self.make_ls_pred(q_max)

            t = self.line_search(q0, i, val_min, val_max, p_min, p_max)

            thresholds[f] = t
        end_time = timeit.default_timer()

        print 'Threshold finding required %d queries' % self.query_count
        print 'Threshold finding took %d seconds' % (end_time - start_time)
        logging.debug(thresholds)
        bins = {}

        for (f, t) in thresholds.iteritems():
            t.sort()
            bins[f] = [Bin(-1.1, t[0], self.eps)] + \
                      [Bin(t[i-1], t[i], self.eps) for i in range(1, len(t))] \
                      + [Bin(t[-1], 1.1, self.eps)]

        for f in self.input_features:
            if f in bins:
                logging.info('{}: {}'.format(f, bins[f]))

        return bins

    def find_bin_index(self, v, f):
        if v <= self.bins[f][0].lower:
            return 0

        if v > self.bins[f][-1].upper:
            return len(self.bins[f])-1

        for i in range(len(self.bins[f])):
            if self.bins[f][i].contains(v):
                return i
        raise ValueError('{} not found for bins {}'.format(v, self.bins[f]))

    def encode(self, X_input):
        assert self.binning or self.one_hot
        assert X_input.shape[1] == len(self.input_features)

        X_input = X_input.copy()
        X = []

        for i in range(len(self.label_encoders)):
            if isinstance(X_input[0, i], basestring):
                X_input[:, i] = self.label_encoders[i].transform(X_input[:, i])

        X_input = self.encoder.transform(X_input)

        X_input_cat = X_input[:, :-len(self.bins)]
        X_input_cont = X_input[:, -len(self.bins):]

        numeric_features = [f for (f, t)
                            in zip(self.input_features, self.feature_types)
                            if t == "NUMERIC"]

        for col, f in enumerate(numeric_features):
            temp = np.zeros((len(X_input_cont), len(self.bins[f])))
            idx = [self.find_bin_index(x, f) for x in X_input_cont[:, col]]
            temp[np.arange(len(X_input)), idx] = 1
            X.append(temp)

        X = np.hstack(X)
        X = np.hstack((X_input_cat, X))

        assert X.shape[1] == self.num_features()
        return X

    def decode(self, x):
        vals = []
        idx = 0
        for (i, f) in enumerate(self.input_features):
            if self.feature_types[i] == "CATEGORICAL":
                l = list(x[self.encoder.feature_indices_[i]:self.encoder.
                         feature_indices_[i+1]])
                assert sum(l) == 1 or sum(l) == 0

                if sum(l) == 0:
                    val = None
                else:
                    val = l.index(1)
                    val = self.label_encoders[i].inverse_transform(val)

                incr = self.encoder.n_values_[i]
            else:
                l = list(x[idx:idx+len(self.bins[f])])
                assert sum(l) == 1 or sum(l) == 0

                if sum(l) == 0:
                    val = None
                else:
                    bin = self.bins[f][l.index(1)]
                    val = bin.repr_val()[0]

                incr = len(self.bins[f])
            vals.append(val)
            idx += incr
        return vals

    def random_input(self, test_size=1):

        if not self.one_hot:
            X = np.zeros((test_size, len(self.input_features)))
        else:
            X = np.empty((test_size, len(self.input_features)), dtype=object)

        for i, f in enumerate(self.input_features):
            if self.feature_types[i] == "CATEGORICAL":
                # choose random values for the categorical features
                X[:, i] = np.random.choice(self.label_encoders[i].classes_,
                                           size=test_size)
            else:
                if self.binning:
                    # choose uniformly random values for the continuous features
                    X[:, i] = [self.bins[f][j].unif_val()
                               for j in np.random.choice(len(self.bins[f]),
                                                         size=test_size)]
                else:
                    X[:, i] = utils.gen_query_set(1, test_size)[:, 0]
        return X

    def gen_query_set(self, n, test_size, force_input_space=False):
        if force_input_space or len(self.input_features) == n:
            X = self.random_input(test_size)
            if self.binning:
                r = -decimal.Decimal(str(self.eps)).as_tuple().exponent
                for i, t in enumerate(self.feature_types):
                    if t == "NUMERIC":
                        X[:, i] = np.round(X[:, i].astype(np.float), r)
            return X

        if not self.incomplete:
            rank = 0
            temp = np.hstack((self.encode(self.random_input()), [[1]]))

            counter = 0
            while rank < min(test_size, self.num_features()+1) \
                    and counter < 10*self.num_features():

                row = np.hstack((self.encode(self.random_input()), [[1]]))

                new_rank = np.linalg.matrix_rank(np.vstack((temp, row)))
                if new_rank > rank:
                    rank = new_rank
                    temp = np.vstack((temp, row))
                counter += 1

            temp = temp[:, :-1]
        else:
            temp = np.eye(self.num_features(), self.num_features())
            temp = np.vstack((temp, np.zeros((1, self.num_features()))))

        if len(temp) < test_size:
            temp = np.vstack((temp, self.encode(
                self.random_input(test_size - len(temp)))))

        assert temp.shape == (test_size, self.num_features())
        return temp

    def prepare_query(self, x):
        if len(x) != len(self.input_features):
            x = self.decode(x)

        feature_vector = dict(zip(self.input_features, x))
        feature_vector = {k: str(v) for (k, v) in feature_vector.iteritems()
                          if v is not None}

        return feature_vector

    def aws_query(self, X):
        if len(X) > 1:
            print 'Current query count: %d' % self.query_count
            print 'Sending {} queries to the oracle'.format(len(X))

        responses = np.array([None]*len(X))

        X = [self.prepare_query(x) for x in X]

        X_cache = [(i, x) for (i, x) in enumerate(X)
                   if hash(frozenset(x.items())) in self.query_cache]

        X_query = [(i, x) for (i, x) in enumerate(X)
                   if hash(frozenset(x.items())) not in self.query_cache]

        print '{} queries can be served from cache'.format(len(X_cache))
        print '{} queries to be sent to AWS'.format(len(X_query))

        for (i, x) in X_cache:
            qhash = hash(frozenset(x.items()))
            responses[i] = self.query_cache[qhash]

        """
        with open('temp.txt', 'w+') as f:
            for (_, x) in X_query:
                f.write('None,' + ','.join(x.values()) + '\n')
        """

        for (i, x) in X_query:
            qhash = hash(frozenset(x.items()))

            response = self.client.predict(
                MLModelId=self.model['MLModelId'],
                Record=x,
                PredictEndpoint=
                'https://realtime.machinelearning.us-east-1.amazonaws.com'
            )['Prediction']

            self.query_count += 1
            self.query_cache[qhash] = response
            responses[i] = response

        return responses

    def query_probas(self, X):
        start_time = timeit.default_timer()

        responses = self.aws_query(X)

        probas = np.zeros((len(X), len(self.get_classes())))

        for i in range(len(X)):
            scores = responses[i]['predictedScores']
            if len(self.get_classes()) == 2:
                class_probas = [1-scores.values()[0], scores.values()[0]]

            else:
                class_probas = [scores[c] for c in self.get_classes()]
            probas[i] = class_probas

        end_time = timeit.default_timer()

        if len(X) > 1:
            print 'Batch of %d queries took %d seconds' \
                  % (len(X), end_time-start_time)

        return probas

    def query(self, X):
        start_time = timeit.default_timer()

        responses = self.aws_query(X)
        labels = []

        for i in range(len(X)):
            label = str(responses[i]['predictedLabel'])
            labels.append(label)

        end_time = timeit.default_timer()

        if len(X) > 1:
            print 'Batch of %d queries took %d seconds' \
                  % (len(X), end_time-start_time)

        return np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='a dataset')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--incomplete', dest='incomplete',
                        action='store_true', help='allow incomplete queries')
    args = parser.parse_args()

    dataset = args.data
    seed = args.seed
    incomplete = args.incomplete
    verbose = args.verbose

    if verbose:
        level = logging.INFO

        logger = logging.getLogger()
        logger.setLevel(level)
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    np.random.seed(seed)

    _, _, X, _, _ = utils.prepare_data(dataset, onehot=False, labelEncode=False)

    cat_idx = [i for i in range(len(X.columns))
               if isinstance(X.iloc[0][i], basestring)]
    cont_idx = range(X.shape[1])
    for i in cat_idx:
        cont_idx.remove(i)
    X = X[cat_idx + cont_idx].values

    ext = AWSRegressionExtractor(dataset, X.copy(), cat_idx,
                                 incomplete=incomplete)

    try:
        X_test = X[0:500]

        if ext.binning:
            r = -decimal.Decimal(str(ext.eps)).as_tuple().exponent
            for i, t in enumerate(ext.feature_types):
                if t == "NUMERIC":
                    X_test[:, i] = np.round(X_test[:, i].astype(np.float), r)
    except ValueError:
        X_test = None

    ext.run(args.data, X_test, 500, random_seed=seed,
            alphas=[1], methods=['passive'], baseline=False)

if __name__ == "__main__":
    main()
