from regression_stealer import *
import numpy as np
import sys
from pylearn2.utils import serial


class MLInversionExtractor(RegressionExtractor):
    """
    Extractor for ML Inversion attack.
    """
    def __init__(self, model_file):

        # open a locally stored logistic regression model
        self.model = serial.load(model_file)
        self.multinomial = True
        self.w, self.intercept = self.get_coefficients()
        self.use_intercept = True
        RegressionExtractor.__init__(self)

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
        return predict_classes(X, self.w, self.intercept, self.classes)


def main(argv):
    model_file = argv[0]
    ext = MLInversionExtractor(model_file)
    ext.run(model_file, X_test=None, test_size=10000, 
            alphas=[0.1], methods=['passive'], baseline=False)

if __name__ == "__main__":
    main(sys.argv[1:])
