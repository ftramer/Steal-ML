#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import copy
import decimal


class EqMixin(object):
    """
    Equality wrapper for static objects
    """
    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(frozenset(self.__dict__.iteritems()))


class LeafID(EqMixin):
    """
    A leaf identity
    """
    def __init__(self, val, conf, rounding, path):
        self.val = val
        self.conf = conf

        self.path = frozenset(path)
        if rounding is not None:
            if isinstance(conf, list):
                self.conf = [round(x, rounding) for x in self.conf]
            else:
                self.conf = round(self.conf, rounding)

        if isinstance(self.conf, list):
            self.conf = '-'.join(map(str, self.conf))

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.val == other.val and self.conf == other.conf)

    def __hash__(self):
        return hash(frozenset([self.val, self.conf]))

    def __repr__(self):
        return "ID({}, {}, {})".format(self.val, self.conf, self.path)


class Predicate(EqMixin):
    """
    An abstract splitting predicate
    """
    def __init__(self):
        return


class CAT(Predicate):
    """
    A split on a categorical feature into disjoint sets
    """
    def __init__(self, feature, pos_set, neg_sets):
        self.feature = feature
        self.pos_set = pos_set
        self.neg_sets = neg_sets

    def invert(self):
        inv = []
        for neg_set in self.neg_sets:
            inv.append(CAT(self.feature, neg_set, None))
        return inv

    def __repr__(self):
        if len(self.pos_set) == 1:
            return "{} = {}".format(self.feature.name, self.pos_set[0])
        elif len(self.pos_set) < \
                len(set(self.feature.vals) - set(self.pos_set)):
            return "{} in {}".format(self.feature.name, self.pos_set)
        else:
            return "{} not in {}".\
                format(self.feature.name,
                       list(set(self.feature.vals) - set(self.pos_set)))

    def apply(self, query):
        new_query = copy(query)
        new_query[self.feature.name] = self.pos_set[0]
        return new_query

    def is_valid(self, val):
        return val in self.pos_set or str(val) in self.pos_set

    def __hash__(self):
        return hash(frozenset([self.feature, frozenset(self.pos_set)]))
    
    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.feature == other.feature
            and set(self.pos_set) == set(other.pos_set))


class Comp(Predicate):
    """
    An abstract comparison predicate
    """
    @staticmethod
    def create(feature, val, t, epsilon):
        if val <= t:
            return LTE(feature, t, epsilon)
        else:
            return GT(feature, t, epsilon)


class LTE(Comp):
    """
    Predicates of the form 'value <= threshold'
    """
    def __init__(self, feature, t, epsilon):
        self.feature = feature
        self.t = t
        self.epsilon = epsilon

    def invert(self):
        return [GT(self.feature, self.t, self.epsilon)]

    def __repr__(self):
        return "{} <= {}".format(self.feature.name, self.t)

    def apply(self, query):
        new_query = copy(query)
        new_query[self.feature.name] = self.t
        return new_query

    def is_valid(self, val):
        return val <= self.t


class GT(Comp):
    """
    Predicates of the form 'value > threshold'
    """
    def __init__(self, feature, t, epsilon):
        self.feature = feature
        self.t = t
        self.epsilon = epsilon

    def invert(self):
        return [LTE(self.feature, self.t, self.epsilon)]

    def __repr__(self):
        return "{} > {}".format(self.feature.name, self.t)

    def apply(self, query):
        new_query = copy(query)
        prec = -decimal.Decimal(str(self.epsilon)).as_tuple().exponent
        new_query[self.feature.name] = round(self.t + self.epsilon, prec)
        return new_query

    def is_valid(self, val):
        return val > self.t


def predicate_names(path):
    return set([pred.feature.name for pred in path])
