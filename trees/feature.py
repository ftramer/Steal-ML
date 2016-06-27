#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

UNKNOWN = ['\N', 'Unknown', 'Not Reported']


class Feature(object):
    """
    A Feature used in a decision tree
    """
    def __init__(self, name, key):
        self.name = name
        self.key = key


class ContFeature(Feature):
    """
    A continuous feature
    """
    def __init__(self, name, key, min_val, max_val):
        Feature.__init__(self, name, key)
        self.min_val = min_val
        self.max_val = max_val

    def init_val(self):
        return self.min_val

    def rand_val(self, r=5):
        return np.round(np.random.uniform(self.min_val, self.max_val), r)

    def __repr__(self):
        return "{}: ([{}, {}])".format(self.name, self.min_val, self.max_val)


class CatFeature(Feature):
    """
    A categorical feature
    """
    def __init__(self, name, key, vals):
        Feature.__init__(self, name, key)
        self.vals = vals

    def init_val(self):
        return list(set(self.vals) - set(UNKNOWN))[0]

    def rand_val(self):
        return np.random.choice(list(set(self.vals) - set(UNKNOWN)))

    def __repr__(self):
        return "{}: ([{} - {}])".format(self.name, self.vals[0], self.vals[-1])
