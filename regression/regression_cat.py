from regression_stealer import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import argparse
import sys


class LocalRegressionExtractor(RegressionExtractor):
    """
    Local logistic regression using the implementation in scikit
    """

    def __init__(self, X, y, multinomial, rounding=None):
        self.input_features = X.columns.values

        X = X.values

        cat_idx = [i for i in range(X.shape[1]) if min(X[:, i]) == 0]
        self.encoder = OneHotEncoder(categorical_features=cat_idx, sparse=False)

        X = self.encoder.fit_transform(X)

        self.features = range(X.shape[1])
        self.rounding = rounding

        # train a model on the whole dataset
        self.model = LogisticRegression()
        self.model.fit(X, y)

        self.w = self.model.coef_
        self.intercept = self.model.intercept_
        self.multinomial = multinomial
        assert not (multinomial and len(self.get_classes()) == 2)

        RegressionExtractor.__init__(self)

    def num_features(self):
        return len(self.features)

    def num_input_features(self):
        return len(self.input_features)

    def get_classes(self):
        return self.model.classes_

    def query_probas(self, X):
        if X.shape[1] == self.num_input_features():
            X = self.encode(X)

        p = predict_probas(X, self.w, self.intercept,
                           multinomial=self.multinomial)
        if self.rounding is None:
            return p
        else:
            p = np.round(p, self.rounding)
            return p / np.sum(p, axis=1)[:, np.newaxis]

    def query(self, X):
        if X.shape[1] == self.num_input_features():
            X = self.encode(X)

        return predict_classes(X, self.w, self.intercept, self.classes)

    def random_input(self, test_size=1):
        cat_idxs = self.encoder.categorical_features
        n = self.num_input_features()

        X = np.zeros((test_size, n))
        for i in range(n):
            if i in cat_idxs:
                cat_idx = cat_idxs.index(i)
                # choose random values for the categorical features
                X[:, i] = np.random.choice(self.encoder.n_values_[cat_idx],
                                           size=test_size)
            else:
                # choose uniformly random values for the continuous features
                X[:, i] = utils.gen_query_set(1, test_size)[:, 0]
        return X

    def encode(self, X):
        assert X.shape[1] == self.num_input_features()
        return self.encoder.transform(X)

    def gen_query_set(self, n, test_size, force_input_space=False):

        if force_input_space:
            return self.random_input(test_size)

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

        if len(temp) < test_size:
            temp = np.vstack((temp, self.encode(
                self.random_input(test_size - len(temp)))))

        assert len(temp) == test_size

        return temp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='a dataset')
    parser.add_argument('--multinomial', dest='multinomial',
                        action='store_true', help='multinomial softmax flag')
    parser.add_argument('--rounding', type=int, help='rounding digits')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    data = args.data
    seed = args.seed

    print >> sys.stderr, 'Data: {}, Seed: {}'.format(data, seed)

    np.random.seed(0)
    X_train, y_train, X_test, y_test, _ = utils.prepare_data(data, onehot=False)

    ext = LocalRegressionExtractor(X_train, y_train,
                                   multinomial=args.multinomial,
                                   rounding=args.rounding)

    y_pred = ext.query(ext.encode(X_test))
    print 'training accuracy: {}'.format(accuracy_score(y_test, y_pred))
    print Counter(y_pred)
    print Counter(y_test)

    ext.run(data, X_test, random_seed=seed,
            methods=['passive', 'adapt-local'], baseline=True)

    
if __name__ == "__main__":
    main()
