__author__ = 'fan'

from utils.logger import *
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from scipy.spatial import distance
import numpy as np
import sys
import os

import matplotlib as mpl
mpl.use('Agg')
from collections import deque


class OnlineBase(object):
    def __init__(self, name, label_p, label_n, clf1, n_features, ftype, error):
        assert ftype in ('uniform', 'norm', 'binary')
        self.name = name
        self.clf1 = clf1

        self.NEG, self.POS = label_n, label_p
        self.q = 0

        self.e = error
        self.n_features = n_features
        self.ftype = ftype

        # budget
        self.budget = -1

        # extraction results. Pure python lists XXX
        self.pts_near_b = []
        self.pts_near_b_labels = []

    def set_budget(self, b):
        self.budget = b

    def add_budget(self, b):
        if self.budget == -1:
            self.set_budget(b)
        else:
            self.budget += b

    def random_vector(self, length, label=None, spec=None):
        if spec is not None:
            self.ftype = spec.type
            mean = spec.mean
            low = spec.range[0]
            high = spec.range[1]
        else:
            mean = 0
            low = -1
            high = +1

        if label is not None:
            assert label in (self.NEG, self.POS), 'unknown label %d' % label

        def rv_bi(size):
            a = 2 * np.random.randint(2, size=(size,)) - 1
            return a

        def rv_norm(size):
            if spec is not None and spec.type == 'norm':
                assert len(spec.mean) == self.n_features
                r = np.zeros(size)
                for i in range(0, size):
                    r[i] = np.random.normal(loc=spec.mean[i])
                return r
            else:
                return np.random.normal(loc=mean, size=size)

        def rv_uni(size):
            return np.random.uniform(low, high, size)

        if self.ftype not in ('norm', 'uniform', 'binary'):
            assert False, 'unknown feature type'

        if self.ftype == 'binary':
            rv_gen = rv_bi
        elif self.ftype == 'norm':
            rv_gen = rv_norm
        elif self.ftype == 'uniform':
            rv_gen = rv_uni
        else:
            assert False, 'unknown feature type'

        if label is not None:
            while True:
                a = rv_gen(length)
                l = self.query(a)
                if l == label:
                    return a
                else:
                    logger.debug('Want %d got %d', label, l)
        else:
            return rv_gen(length)

    def query(self, x, count=True):
        if count:
            self.q += 1
            if self.q > 0 and self.q % 100 == 0:
                logger.debug('%d queries consumed' % self.q)

            if self.budget != -1 and self.q > self.budget:
                raise RunOutOfBudget

        if hasattr(self.clf1, 'predict'):
            label = self.clf1.predict(x)
        else:
            label = self.clf1(x)

        return label

    def push_to_b(self, xn, xp, e):
        assert self.query(xn, count=False) == self.NEG
        assert self.query(xp, count=False) == self.POS

        d = distance.euclidean(xn, xp) / \
            distance.euclidean(np.ones(self.n_features), np.zeros(self.n_features))
        if d < e:
            logger.debug('bin search done with %f', d)
            return xn, xp

        mid = .5 * np.add(xn, xp)
        try:
            l = self.query(mid)
            if l == self.NEG:
                return self.push_to_b(mid, xp, e)
            else:
                return self.push_to_b(xn, mid, e)
        except RunOutOfBudget:
            logger.debug('Run out of budget %d, push_to_b failed' % self.budget)
            raise RunOutOfBudget

    def collect_pts(self, n, budget=-1):
        self.set_budget(budget)
        if n == -1:
            assert budget > 0, 'exhaust without budget is doomed.'
            exhaust = True
        else:
            exhaust = False
        if n > 0 and n % 2 != 0:
            logger.debug('n should be even, got %d' % n)
            n += 1

        m = int(n / 2)
        if exhaust:
            m = sys.maxint
        pts_near_b_in_x = []
        pts_near_b_in_x_label = []
        for i in xrange(0, m):
            try:
                # get a pair with different labels
                x1 = self.random_vector(self.n_features, self.NEG, None)
                x2 = self.random_vector(self.n_features, self.POS, None)
                logger.debug('%s feature no. %d found', self.ftype, i)

                q = self.q
                xb1, xb2 = self.push_to_b(x1, x2, self.e)
                logger.debug('push consumed %d queries' % (self.q - q))
            except RunOutOfBudget:
                logger.debug('Run out of budget, collecting point has to stop. %d pairs collected', i)
                break
            logger.debug('boundary pair no. %d found', i)

            pts_near_b_in_x.extend((xb1, xb2))
            pts_near_b_in_x_label.extend((self.NEG, self.POS))
            # pts_near_b_in_x.extend((x1, x2))
            # pts_near_b_in_x_label.extend((self.NEG, self.POS))

        self.pts_near_b = pts_near_b_in_x
        self.pts_near_b_labels = pts_near_b_in_x_label
        # self.dump_queries()
        return pts_near_b_in_x, pts_near_b_in_x_label

    def random_vector_n_pairs(self, n, length):
        """
        generates n pairs of random vectors. Efficient.
        :return:
        """

        negs, poss = [], []
        n_n, n_p = 0, 0
        label = None
        while True:
            try:
                t = self.random_vector(length, label=label)
                l = self.query(t)
            except RunOutOfBudget:
                logger.warn('Run out of budget. %d negs and %d poss collected.', len(negs), len(poss))
                break
            if label is not None and l != label:
                continue
            if l == self.NEG:
                negs.append(t)
                n_n += 1
                if n_n == n:
                    if label is not None:
                        break
                    label = self.POS
            else:
                poss.append(t)
                n_p += 1
                if n_p == n:
                    if label is not None:
                        break
                    label = self.NEG

        return negs, poss

    def collect_one_pair(self):
        x1 = self.random_vector(self.n_features)
        if self.clf1(x1) == self.NEG:
            next = self.POS
        else:
            next = self.NEG

        self.set_budget(50)
        try:
            x2 = self.random_vector(self.n_features, next)
            return self.push_to_b(x1, x2, self.e)
        except RunOutOfBudget:
            return [x1]

    def collect_up_to_budget(self, budget):
        self.set_budget(budget)
        neg_q = deque()
        pos_q = deque()
        while True:
            try:
                x1 = self.random_vector(self.n_features)
                if self.query(x1) == self.NEG:
                    neg_q.append(x1)
                else:
                    pos_q.append(x1)

                while len(neg_q) > 0 and len(pos_q) > 0:
                    xp = pos_q.popleft()
                    xn = neg_q.popleft()
                    self.pts_near_b.extend(self.push_to_b(xn, xp, self.e))
                    self.pts_near_b_labels.extend((self.NEG, self.POS))
            except RunOutOfBudget:
                logger.debug('Run out of budget (%d), %d pairs collected so far' % \
                      (self.budget, len(self.pts_near_b)/2))
                break

    def collect_pts_frugal(self, n, budget=-1):
        self.set_budget(budget)
        if n == -1:
            assert budget > 0, 'exhaust without budget is doomed.'
            exhaust = True
        else:
            exhaust = False

        if n > 0 and n % 2 != 0:
            logger.warn('n should be even, got %d' % n)
            n += 1

        pts_near_b_in_x = []
        pts_near_b_in_x_label = []

        from math import sqrt, ceil

        negs, poss = self.random_vector_n_pairs(ceil((sqrt(n))), self.n_features)

        n_got = 0
        if exhaust:
            n = sys.maxint
        try:
            for nn in negs:
                for pp in poss:
                    if n_got >= n:
                        break
                    xb1, xb2 = self.push_to_b(nn, pp, self.e)
                    pts_near_b_in_x.extend((xb1, xb2))
                    pts_near_b_in_x_label.extend((self.NEG, self.POS))
                    n_got += 2
        except RunOutOfBudget:
            logger.error('Run out of budget, collecting point has to stop. %d pairs collected', n_got)

        self.pts_near_b = pts_near_b_in_x
        self.pts_near_b_labels = pts_near_b_in_x_label
        # self.dump_queries()
        return pts_near_b_in_x, pts_near_b_in_x_label

    # def dump_queries(self):
    #     import pickle
    #
    #     if not os.path.isdir(self.dump_dir):
    #         os.mkdir(self.dump_dir)
    #
    #     suffix = ('e', 'b', 's', 'q')
    #     values = (self.e, len(self.pts_near_b), len(self.support_labels), self.q)
    #
    #     name = ''.join(np.core.defchararray.add(suffix, map(str, values)).tolist())
    #     name += '.query'
    #     dump_name = os.path.join(self.dump_dir, name)
    #
    #     with open(dump_name, 'wb') as dumpfile:
    #         pickle.dump((self.pts_near_b, self.pts_near_b_labels,
    #                      self.support_pts, self.support_labels),
    #                     dumpfile)
    #
    #     logger.info('Queries dumped into %s' % dump_name)

    def get_n_query(self):
        return self.q

    def get_QSV(self):
        return self.pts_near_b

    def get_QSV_labels(self):
        return self.pts_near_b_labels


class RunOutOfBudget(Exception):
    pass


class FeatureSpec(object):
    def __init__(self, type, range, mean):
        self.type = type
        self.range = range
        self.mean = mean
