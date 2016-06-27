__author__ = 'Fan'

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../libsvm-3.20/python'))

from algorithms.OnlineBase import OnlineBase
from svmutil import svm_load_model, svm_predict


class LibSVMOnline(OnlineBase):
    def __init__(self, name, model, labels, n_features, ftype, error):
        _p, _n = labels
        super(self.__class__, self).__init__(name, _p, _n, None, n_features, ftype, error)
        self.model = svm_load_model(model)

        def clf(x):
            if hasattr(x, 'tolist'):
                xx = x.tolist()
            else:
                xx = x
            p_label, _, _ = svm_predict([0], [xx], self.model, '-q')
            return p_label[0]

        self.clf1 = clf

    def batch_predict(self, xs, count=True):
        if hasattr(xs, 'tolist'):
            xs = xs.tolist()
        xs = [ x.tolist() if hasattr(x, 'tolist') else x for x in xs]
        p_label, _, _ = svm_predict([0]*len(xs), xs, self.model, '-q')

        if count:
            self.q += len(p_label)
        return map(int, p_label)