#!/usr/bin/env python
# -*- coding: utf-8 -*-

from predicate import *
from feature import *
import abc
import logging
from logging import DEBUG


def make_query(query, feature_name, val):
    """
    Modify a given query by switching the value of one feature
    """
    new_query = copy(query)
    new_query[feature_name] = val
    return new_query


def pred_match(query, path):
    for (featureName, val) in query.iteritems():
        for pred in path:
            if pred.feature.name == featureName:
                if not pred.is_valid(val):
                    #print 'val={} is not valid for predicate {}'.format(val, pred)

                    # the query does not satisfy this set of predicates
                    return False
    return True


def pred_valid(query, pred_list):
    """
    Check if a query satisfies any of the predicates in a list
    """
    for predicates in pred_list:
        if pred_match(query, predicates):
            return True

    # the query satisfied none of the predicate sets
    return False


class TreeExtractor(object):
    """
    Algorithm for extracting a decision tree from black-box queries
    """
    def __init__(self, epsilon=0.01, rounding=None):
        self.features = self.get_features()
        self.queryCache = {}
        self.queryCount = 0
        self.rounding = rounding
        self.epsilon = epsilon

    def extract(self, incomplete_queries=False):
        """
        Starts from an arbitrary query and recursively extracts the tree
        """
        all_leaves = {}
        query_pool = []

        query0 = {feature.name: feature.rand_val() for feature in self.features}
        query_pool += [(query0, [])]

        logging.log(DEBUG, 'initial query pool: {}'.format(query_pool))

        while query_pool:
            (query, query_pred) = query_pool.pop()
            leaf_id = self.predict(query)

            logging.log(DEBUG, 'leaf query {} got {}'.format(query, leaf_id))
            logging.log(DEBUG, 'query_pred: {}'.format(query_pred))

            # check if we already got the same leaf identifier with a coherent
            # query
            if leaf_id in all_leaves and pred_valid(query, all_leaves[leaf_id]):
                logging.log(DEBUG, 'leaf already visited')
            else:
                logging.log(DEBUG, 'New leaf found! Looking for predicates')

                # get the predicates for this leaf, and the new queries to add
                # to the pool
                predicates, new_queries = \
                    self.get_predicates(query, leaf_id, query_pred,
                                        epsilon=self.epsilon,
                                        incomplete_queries=incomplete_queries)

                # if we are unlucky, two different leaves will have the same ID
                # we might detect this if the current query doesn't match the
                # previous path
                if leaf_id in all_leaves:
                    logging.log(DEBUG, "TWO DIFFERENT LEAVES HAVE SAME ID!!!")
                    all_leaves[leaf_id].append(predicates+query_pred)
                else:
                    all_leaves[leaf_id] = [predicates+query_pred]

                logging.log(DEBUG, 'predicates: {}'.format(predicates))

                for newQuery, pred in new_queries:
                    logging.log(DEBUG,
                                'adding query {} to pool'.format(newQuery))
                    # queryPool += [(newQuery, pred)]
                    query_pool += [(newQuery, [])]
        
        return all_leaves

    def extract_top_down(self):
        """
        Starts from an empty query and reconstructs the tree from the top
        """

        all_nodes = {}
        all_leaves = {}
        query_pool = []

        query0 = {}
        query_pool += [(query0, [])]

        logging.log(DEBUG, 'initial query pool: {}'.format(query_pool))

        while query_pool:
            (query, query_pred) = query_pool.pop(0)
            node_id = self.predict(query)
            logging.log(DEBUG, 'node query {} got {}'.format(query, node_id))
            logging.log(DEBUG, 'query_pred: {}'.format(query_pred))

            if node_id in all_nodes and pred_valid(query, all_nodes[node_id]):
                logging.log(DEBUG, 'node already visited')
            else:
                # First find any predicates we might have missed because we
                # "traversed" a node.
                # Add queries to the pool to visit any missed branches
                missed_preds, temp_pool = \
                    self.top_down_features_test(query, query_pred, node_id,
                                                find_missing=True)
                query_pool += temp_pool

                # Update this node's predicate and look for the next split
                query_pred += missed_preds
                temp_pool = \
                    self.top_down_features_test(query, query_pred, node_id,
                                                find_missing=False)
                
                # no split found means we are at a leaf
                if not temp_pool:
                    if node_id in all_leaves:
                        logging.log(DEBUG,
                                    "TWO DIFFERENT Leaves HAVE SAME ID!!!")
                        all_leaves[node_id].append(query_pred)
                    else:
                        all_leaves[node_id] = [query_pred]

                query_pool += temp_pool
                
                logging.log(DEBUG, 'new node found: {} with predicate {}'.
                            format(node_id, query_pred))
                if node_id in all_nodes:
                    logging.log(DEBUG, "TWO DIFFERENT NODES HAVE SAME ID!!!")
                    all_nodes[node_id].append(query_pred)
                else:
                    all_nodes[node_id] = [query_pred]
        return all_leaves
    
    def top_down_features_test(self, query, query_pred, node_id,
                               find_missing=False):
        """
        Check all features for either missed splits or new splits

        - A missed split is one for which some values stay in the current node,
        implying that the split actually occurs higher up in the tree
        (or it is a partial new split)
        """

        missed_preds = []
        temp_pool = []
        
        if find_missing:
            logging.log(DEBUG, 'Testing all features for missing splits')
        else:
            logging.log(DEBUG, 'Testing all features for new splits')
        
        # check which features we should try to split on
        split_features = node_id.path.difference(set(pred.feature.name
                                                     for pred in query_pred))
        if not split_features:
            split_features = [f.name for f in self.features]
        logging.log(DEBUG, 'split features: {}'.format(split_features))

        for feature in self.features:
            
            # only look for missing features in the ones that appeared at
            # least once on the path
            if find_missing and not [pred for pred in query_pred
                                     if pred.feature == feature]:
                continue

            # only consider splitting features that appear in the "fields"
            # information
            if not find_missing and feature.name not in split_features:
                continue

            logging.log(DEBUG, 'Testing feature "{}"'.format(feature.name))
            new_queries = []
            feature_missed_preds = []
                
            if isinstance(feature, ContFeature):
                
                # find splitting thresholds
                (min_val, max_val) = self.get_bounds(feature, query_pred)
                thresholds = \
                    self.test_cont_feature(query, feature, self.epsilon,
                                           min_val=min_val, max_val=max_val)

                if thresholds:
                    # add new queries to explore new paths
                    for i in range(len(thresholds)):
                        pred = LTE(feature, thresholds[i], self.epsilon)
                        new_query = pred.apply(query)
                        new_id = self.predict(new_query)
                        if new_id == node_id:
                            logging.log(DEBUG, 'pred {} missed'.format([pred]))
                            feature_missed_preds += [pred]
                        else:
                            logging.log(DEBUG, 'adding query {} to pool'.
                                        format(new_query))
                            new_queries += [(new_query, query_pred+[pred])]

                    pred = GT(feature, thresholds[-1], self.epsilon)
                    new_query = pred.apply(query)
                    new_id = self.predict(new_query)
                    if new_id == node_id:
                        logging.log(DEBUG, 'pred {} missed'.format(pred))
                        feature_missed_preds += [pred]
                    else:
                        logging.log(DEBUG, 'adding query {} to pool'.
                                    format(new_query))
                        new_queries += [(new_query, query_pred+[pred])]

            elif isinstance(feature, CatFeature):
                # find splitting categories
                categories = self.get_categories(feature, query_pred)
                
                if len(categories) > 1:
                    cat_ids = self.test_cat_feature(query, feature,
                                                   categories=categories)
                    
                    if len(cat_ids) > 1:
                        # add new queries to explore new paths
                        for vals in sorted(cat_ids.values(),
                                           key=len, reverse=False):
                            pred = CAT(feature, vals, None)
                            
                            new_query = pred.apply(query)
                            new_id = self.predict(new_query)
                            if new_id == node_id:
                                feature_missed_preds += [pred]
                            else:
                                logging.log(DEBUG, 'adding query {} to pool'.
                                            format(new_query))
                                new_queries += [(new_query, query_pred+[pred])]

            if not find_missing or feature_missed_preds:
                temp_pool += new_queries
                missed_preds += feature_missed_preds
        
        if find_missing:    
            logging.log(DEBUG, 'missed_preds: {}'.format(missed_preds))
            return missed_preds, temp_pool
        else:
            return temp_pool

    def get_predicates(self, query, leaf_id, query_pred,
                       incomplete_queries=False, epsilon=0.01):
        """
        Finds all predicates that are satisfied by a leaf and generate new
        queries to  potentially undiscovered leaves
        """
        predicates = []
        queries = []
        for feature in self.features:
            logging.log(DEBUG, 'Testing feature "{}"'.format(feature.name))

            if incomplete_queries:
                # test if removing the feature leads to a different node
                incomplete_query = copy(query)
                del incomplete_query[feature.name]
                incomplete_id = self.predict(incomplete_query)

            if not incomplete_queries or incomplete_id != leaf_id:
                if isinstance(feature, ContFeature):
                    # find splitting thresholds
                    (min_val, max_val) = self.get_bounds(feature, query_pred)
                    thresholds = self.test_cont_feature(query, feature, epsilon,
                                                        min_val=min_val,
                                                        max_val=max_val)

                    predicates += [Comp.create(feature, query[feature.name], t,
                                               epsilon) for t in thresholds]
                    
                    if thresholds:
                        if query[feature.name] > thresholds[0]:
                            pred = LTE(feature, thresholds[0], self.epsilon)
                            new_query = pred.apply(query)
                            queries += [(new_query, [pred])]

                        for i in range(len(thresholds) - 1):
                            if query[feature.name] <= thresholds[i] \
                                    or query[feature.name] > thresholds[i + 1]:
                                pred1 = GT(feature, thresholds[i], self.epsilon)
                                pred2 = LTE(feature, thresholds[i + 1],
                                            self.epsilon)
                                new_query = pred1.apply(query)
                                queries += [(new_query, [pred1, pred2])]

                        if query[feature.name] <= thresholds[-1]:
                            pred = GT(feature, thresholds[-1], self.epsilon)
                            new_query = pred.apply(query)
                            queries += [(new_query, [pred])]

                elif isinstance(feature, CatFeature):
                    # find mapping from categories to leaves
                    categories = self.get_categories(feature, query_pred)
                    if len(categories) > 1:
                        cat_ids = self.test_cat_feature(query, feature,
                                                        categories=categories)
                        
                        if len(cat_ids) > 1:
                            pos_set = cat_ids[leaf_id]
                            del cat_ids[leaf_id]
                            pred = CAT(feature, pos_set, cat_ids.values())
                            predicates += [pred]
                            for invPred in pred.invert(): 
                                new_query = invPred.apply(query)
                                queries += [(new_query, [invPred])]

        return predicates, queries

    def test_cont_feature(self, query, feature, epsilon, min_val=None,
                          max_val=None):
        """
        FInd splits on a continuous feature
        """
        if min_val is None:
            min_val = feature.min_val
        if max_val is None:
            max_val = feature.max_val

        query_max = make_query(query, feature.name, max_val)
        max_id = self.predict(query_max)
        query_min = make_query(query, feature.name, min_val)
        min_id = self.predict(query_min)
        logging.log(DEBUG, '\tmin val {} got {}'.format(min_val, min_id))
        logging.log(DEBUG, '\tmax val {} got {}'.format(max_val, max_id))
        
        # search for any splitting thresholds
        thresholds = sorted(self.line_search(query, feature.name, min_val,
                                             max_val, min_id, max_id, epsilon))
        logging.log(DEBUG, '\tthresholds: {}'.format(thresholds))
        return thresholds

    def test_cat_feature(self, query, feature, categories=None):
        """
        Find splits on a categorical feature
        """
        
        if not categories:
            categories = feature.vals

        # map of a leaf's ID to all the values that lead to it
        cat_ids = {}

        for val in categories:
            # test each value one after the other
            query_cat = make_query(query, feature.name, val)
            cat_id = self.predict(query_cat)
            logging.log(DEBUG, '\t val {} got {}'.format(val, cat_id))

            if cat_id in cat_ids:
                cat_ids[cat_id].append(val)
            else:
                cat_ids[cat_id] = [val]

        return cat_ids

    @abc.abstractmethod
    def get_features(self):
        """
        Parse the tree model to get all the features
        """

    @abc.abstractmethod
    def make_prediction(self, query):
        """
        Issue a query to the black-box model
        """

    def predict(self, query):
        """
        Issue a query to the black-box model
        """
        # check if the same query was already answered
        qhash = hash(frozenset(query.items()))
        if qhash in self.queryCache:
            return self.queryCache[qhash]

        res_id = self.make_prediction(query)

        self.queryCount += 1
        self.queryCache[qhash] = res_id

        return res_id

    def eps_round(self, x):
        """
        Round a floating point value to the nearest multiple of eps
        """
        return round(x / self.epsilon) * self.epsilon

    def line_search(self, query, feature, min_val, max_val, min_id,
                    max_id, epsilon):
        """
        Perform a line search on a continuous feature to find all splitting
        thresholds
        """
        prec = -decimal.Decimal(str(epsilon)).as_tuple().exponent

        def int_line_search(query, feature, min_mult, max_mult, min_id, max_id):
            if min_id == max_id:
                return []

            # if we've narrowed down the split to less than epsilon,
            # call it a day
            if (max_mult - min_mult) <= 1:
                threshold = round(min_mult * epsilon, prec)
                logging.log(DEBUG, 'ending line search with {} ({} - {}) for '
                                   'initial query {}'.
                            format(threshold, min_mult, max_mult, query))
                return [threshold]

            # split at half way and check recursively on both sides
            split = (min_mult + max_mult)/2
            split_query = make_query(query, feature,
                                     round(split * epsilon, prec))
            split_id = self.predict(split_query)

            logging.log(DEBUG, '\tval {} got {}'.
                        format(round(split*epsilon, prec), split_id))
            return int_line_search(query, feature, split,
                                   max_mult, split_id, max_id) + \
                   int_line_search(query, feature, min_mult,
                                   split, min_id, split_id)

        max_mult = int(round(max_val / epsilon))
        min_mult = int(round(min_val / epsilon))
        return int_line_search(query, feature, min_mult, max_mult,
                               min_id, max_id)

    def get_bounds(self, feature, pred_list, exact=False):
        """
        Get the bounds for a continuous feature from a list of predicates
        """
        min_val = self.eps_round(feature.min_val)
        max_val = self.eps_round(feature.max_val)

        for pred in pred_list:
            if pred.feature.name == feature.name:
                if isinstance(pred, LTE):
                    max_val = min(max_val, pred.t)
                elif isinstance(pred, GT):
                    if exact:
                        min_val = max(min_val, pred.t)
                    else:
                        min_val = max(min_val, pred.t + self.epsilon)
                else:
                    raise ValueError('unknown comparison predicate')

        return min_val, max_val

    def get_categories(self, feature, predicate_list):
        """
        Get all categories that a categorical feature may take to satisfy all
        predicates
        """
        cats = set(feature.vals)

        for pred in predicate_list:
            if pred.feature.name == feature.name:
                if isinstance(pred, CAT):
                    cats = cats.intersection(set(pred.pos_set)) 
                else:
                    raise ValueError('unkown comparison predicate')

        assert len(cats) >= 1
        return list(cats)

    def equal_paths(self, path1, path2):
        """
        Check if two predicate paths are equal
        """
        for feature in self.features:
            if isinstance(feature, ContFeature):
                (min_val1, max_val1) = self.get_bounds(feature, path1)
                (min_val2, max_val2) = self.get_bounds(feature, path2)
                if abs(min_val1 - min_val2) > self.epsilon \
                        or abs(max_val1 - max_val2) > self.epsilon:
                    return False
            else:
                cats1 = self.get_categories(feature, path1)
                cats2 = self.get_categories(feature, path2)
                if not set(cats1) == set(cats2):
                    return False
        return True

    def is_parent(self, path1, path2):

        if path1 == path2:
            return False

        for feature in self.features:
            if isinstance(feature, ContFeature):
                (min_val1, max_val1) = self.get_bounds(feature, path1)
                (min_val2, max_val2) = self.get_bounds(feature, path2)
                if min_val1 > min_val2 or max_val1 < max_val2:
                    return False
            else:
                cats1 = self.get_categories(feature, path1)
                cats2 = self.get_categories(feature, path2)
                if not set(cats2).issubset(set(cats1)):
                    return False
        return True

    def merge_all_leaves(self, leaves):
        """
        Attempt to merge predicate paths that lead to leaves with the same
        identity
        """
        """
        filtered_leaves = {}
        for (id, l1) in leaves.iteritems():
            filtered_leaves[id] = []
            for p1 in l1:
                keep = True
                for l2 in leaves.values():
                    for p2 in l2:
                        if self.is_parent(p1, p2):
                            keep = False
                if keep:
                    filtered_leaves[id].append(p1)
        leaves = filtered_leaves
        """

        merged = {}
        for (leafID, values) in leaves.iteritems():
            if len(values) > 1:
                logging.log(DEBUG, 'merging {}'.format(leafID))
                merged[leafID] = copy(values) + self.merge_all_preds(values)
            else:
                merged[leafID] = values
        return merged
        
    def merge_all_preds(self, preds):
        """
        Attempt to merge predicate paths for a given leaf identity
        """
        merged = []
        
        while preds:
            pred1 = preds.pop()
            found_merge = False
            for pred2 in copy(preds):
                try:
                    pred3 = self.merge_preds(pred1, pred2)
                    logging.log(DEBUG, 'merged to {}'.format(pred3))
                    if pred3 == pred1 or pred3 == pred2:
                        logging.log(DEBUG, 'no new merge...')
                        continue
                    preds += [pred3]
                    found_merge = True
                except ValueError:
                    pass
            
            if not found_merge:
                merged += [pred1]
        
        return merged
    
    def merge_preds(self, pred1, pred2):
        """
        Attempt to merge two predicate paths
        """
        pred3 = []
        
        logging.log(DEBUG, 'trying to merge')
        logging.log(DEBUG, '{}'.format(pred1))
        logging.log(DEBUG, '{}'.format(pred2))
        
        diff = []
        for feature in self.features:
            if isinstance(feature, ContFeature):
                if self.get_bounds(feature, pred1, exact=False) != \
                        self.get_bounds(feature, pred2, exact=False):
                    diff += [feature]
            else:
                if set(self.get_categories(feature, pred1)) \
                        != set(self.get_categories(feature, pred2)):
                    diff += [feature]

        logging.log(DEBUG, 'diff = {}'.format([f.name for f in diff]))

        for feature in self.features:
            logging.log(DEBUG, 'feature = {}'.format(feature.name))
            if isinstance(feature, ContFeature):
                (min_val1, max_val1) = \
                    self.get_bounds(feature, pred1, exact=False)
                (min_val2, max_val2) = \
                    self.get_bounds(feature, pred2, exact=False)
                
                if (min_val1, max_val1) == (feature.min_val, feature.max_val) \
                        and (min_val2, max_val2) == (feature.min_val,
                                                     feature.max_val):

                    continue

                if (min_val1, max_val1) == (feature.min_val, feature.max_val) \
                        or (min_val2, max_val2) == (feature.min_val,
                                                    feature.max_val):

                    logging.log(DEBUG, 'not compatible: feature appears '
                                       'only once')
                    raise ValueError()

                if min_val1 > max_val2 + self.epsilon:
                    logging.log(DEBUG, 'not compatible {} > {}'.
                                format(min_val1, max_val2))
                    raise ValueError()

                elif min_val2 > max_val1 + self.epsilon:
                    logging.log(DEBUG, 'not compatible {} > {}'.
                                format(min_val2, max_val1))
                    raise ValueError()

                else:
                    if len(diff) == 1:
                        min_val = min(min_val1, min_val2)
                        max_val = max(max_val1, max_val2)
                    else:
                        min_val = max(min_val1, min_val2)
                        max_val = min(max_val1, max_val2)

                    logging.log(DEBUG, 'merging to [{}, {}]'.
                                format(min_val, max_val))
                    if min_val == feature.min_val:
                        pred3 += [LTE(feature, max_val, self.epsilon)]
                    elif max_val == feature.max_val:
                        pred3 += [GT(feature, min_val - self.epsilon,
                                     self.epsilon)]
                    else:
                        pred3 += [GT(feature, min_val - self.epsilon,
                                     self.epsilon),
                                  LTE(feature, max_val, self.epsilon)]
                
            else:
                cats1 = self.get_categories(feature, pred1)
                cats2 = self.get_categories(feature, pred2)

                if len(cats1) == len(feature.vals) \
                        and len(cats2) == len(feature.vals):
                    continue

                if len(cats1) == len(feature.vals) \
                        or len(cats2) == len(feature.vals):
                    logging.log(DEBUG, 'not compatible: feature appears '
                                       'only once')
                    raise ValueError()
                
                if len(diff) == 1:
                    cats3 = CAT(feature,
                                list(set(cats1).union(set(cats2))), None)
                elif set(cats1).intersection(set(cats2)):
                    cats3 = CAT(feature,
                                list(set(cats1).intersection(set(cats2))), None)
                else:
                    continue
                logging.log(DEBUG, 'merging to {}'.format(cats3.pos_set))
                pred3 += [cats3]
        
        return pred3

    def generate_uniform_queries(self, n=10000):
        queries = [{feature.name: feature.rand_val()
                    for feature in self.features}
                   for _ in xrange(n)]
        return queries


