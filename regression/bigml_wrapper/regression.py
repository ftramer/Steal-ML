from regression_stealer import *
import json
import numpy as np
import sys

STORAGE = '../data/bigml_models'


class BigMLRegressionExtractor(RegressionExtractor):
    """
    Extractor for regression models trained by BigML.
    BigML uses the one-vs-rest training approach
    """
    def __init__(self, model_id):

        # open a locally stored logistic regression model
        with open(STORAGE + '/logisticregression_%s' % model_id) \
                as data_file:
            self.model = json.load(data_file)

        self.multinomial = False
        self.w, self.intercept = self.get_coefficients()
        self.use_intercept = True
        RegressionExtractor.__init__(self)

    def num_features(self):
        count = 0
        for field in self.model['input_fields']:
            feature = self.model['logistic_regression']['fields'][field]
            if feature['optype'] == 'categorical':
                # categorical features are 'one-hot' encoded as
                # [cat_1, cat_2, ..., cat_m, N.A.]
                count += len(feature['summary']['categories']) + 1
            else:
                assert feature['optype'] == 'numeric', feature['optype']
                count += 1
        return count

    def get_classes(self):
        return np.array([str(c) for (c, _)
                         in self.model['logistic_regression']['coefficients']])

    def get_coefficients(self):
        coeffs = self.model['logistic_regression']['coefficients']

        if len(coeffs) == 2:
            # BigML provides coefficient vectors for both classes but we only
            # need one of them.
            w = np.zeros((1, len(coeffs[0][1])))
            w[0] = coeffs[0][1]
            intercept = w[:, -1]
            w = w[:, :-1]
        else:
            # the intercept is at the end of the coefficient vector
            w = np.zeros((len(coeffs), len(coeffs[0][1])))
            for i in range(w.shape[0]):
                w[i] = coeffs[i][1]

            intercept = w[:, -1]
            w = w[:, :-1]
        return w, intercept

    def query_probas(self, X):
        p = predict_probas(X, self.w, self.intercept,
                           multinomial=self.multinomial)

        # BigML appears to round probabilities to 5 decimal places
        return np.round(p, 5)
        #return p

    def query(self, X):
        return predict_classes(X, self.w, self.intercept, self.classes)

    def make_bigml_query(self):
        # create a dummy query to test whether our local predictions are the
        # same as those returned by BigML
        dict = {}
        vec = np.zeros(self.num_features())
        i = 0
        for field in self.model['input_fields']:
            field = str(field)
            feature = self.model['logistic_regression']['fields'][field]
            if feature['optype'] == 'categorical':
                cats = feature['summary']['categories']
                dict[field] = str(cats[0][0])
                vec[i] = 1

                i += len(cats) + 1
            else:
                dict[field] = 1
                vec[i] = 1
                i += 1

        return dict, vec


def main(argv):
    data = argv[0]

    if data == "iris":
        model_id = "5660065e9ed2332f8100843b"
    elif data == "german":
        model_id = "5667f7d49ed2332f83030279"
    elif data == "spam":
        model_id = "5667fd4e9ed2332f8303029a"
    else:
        raise ValueError("unknown dataset")

    ext = BigMLRegressionExtractor(model_id)

    ext.run(data, X_test=None, test_size=10000)

if __name__ == "__main__":
    main(sys.argv[1:])
