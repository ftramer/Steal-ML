__author__ = 'Fan'

import numpy as np

from OfflineBase import OfflineBase
from utils.logger import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PolySolver(OfflineBase):
    def __init__(self, name, x, y, val_x, val_y, test_x, test_y, fmap, n_features):
        super(self.__class__, self).__init__(x, y, val_x, val_y)

        self.name = name
        self.fmap = fmap
        self.n_features = n_features

        _y = list(set(y))
        _y.sort()
        self.NEG, self.POS = _y

        self.w = None
        self.b = 0

        self.test_x = test_x
        self.test_y = test_y

        self.solve_score = 0

    # def solve_in_f(self):
    #     n_pts = int(2 * len(self.fmap([0] * self.n_features)))
    #     logger.info('requires %d pts' % n_pts)
    #     if len(self.queries) < n_pts:
    #         logger.error("can't do")
    #         return
    #     else:
    #         logger.info('solve with %d points' % len(self.queries))
    #
    #     m = len(self.queries)
    #     para = np.matrix(map(self.fmap, self.queries))
    #
    #     logger.info('done solving')
    #
    #     bb = -1 * np.ones(m).T
    #     self.w, _, _, _ = np.linalg.lstsq(para, bb)
    #     self.b = 1
    #
    #     y_pred = [self.predict(d) for d in self.queries]
    #     train_score = sm.accuracy_score(self.queries_labels, y_pred)
    #     if train_score < .5:
    #         self.w *= -1
    #         self.b *= -1
    #         train_score = 1 - train_score
    #
    #     self.solve_score = train_score
    #
    #     y_pred = [self.predict(d) for d in self.test_x]
    #     test_score = sm.accuracy_score(self.test_y, y_pred)
    #     auc = sm.roc_auc_score(self.test_y, y_pred)
    #
    #     r = Result('poly', 'poly', dict(d=2, gamma=1), train_score, test_score, auc)
    #     return r

    def grid_retrain_in_f(self):
        q = map(self.fmap, self.X_ex)
        if hasattr(self.test_x, 'tolist'):
            test = self.test_x.tolist()
        else:
            test = self.test_x

        if hasattr(self.val_x, 'tolist'):
            val_x = self.val_x.tolist()
        else:
            val_x = self.val_x
        t = map(self.fmap, test)
        v = map(self.fmap, val_x)
        from algorithms.LinearTrainer import LinearTrainer
        l = LinearTrainer(self.name, q, self.y_ex,
                          v, self.val_y,
                          t, self.test_y, self.n_features)

        r = l.grid_search()
        r.kernel = 'poly'
        return r

    def calc_solve_score(self):
        return self.batch_eval(self.X_ex, self.y_ex)

    def calc_test_score(self):
        return self.batch_eval(self.test_x, self.test_y)

    def batch_eval(self, x, y):
        score = 0.0
        yy = [self.predict(d) for d in x]
        for y1, y2 in zip(yy, y):
            if y1 == y2:
                score += 1.0

        score /= float(len(yy))

        return score

    def predict(self, x):
        x = np.ravel(x)
        logger.debug('predicting %s', str(x))
        try:
            xx = self.fmap(x)
        except ValueError:
            print x
            return self.POS
        d = np.inner(xx, self.w) + self.b
        return self.POS if d > 0 else self.NEG
