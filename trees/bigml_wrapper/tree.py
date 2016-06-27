#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from bigml.model import Model
from bigml.api import BigML
import bigml.util as util

from tree_stealer import *
from feature import *
from bigml_wrapper.utils import *

from models import models, black_box_models
from sklearn.metrics import accuracy_score

import logging
import sys
import timeit
import requests
import json


STORAGE = '../data/bigml_models'
BB_KEY = "238d786823e79fe6de425c9732c928ebb86f6351"


class BigMLTreeExtractor(TreeExtractor):

    def __init__(self, data, epsilon=0.01, rounding=None, black_box=False):
        self.black_box = black_box

        if not self.black_box:
            model_id = models[data]
            # retrieve a model from local storage or from bigml.io
            # (only works for public models)
            try:
                self.model = Model('model/{}'.format(model_id),
                                   api=BigML(storage=STORAGE))
            except ValueError:
                self.model = Model('public/model/{}'.format(model_id),
                                   api=BigML(storage=STORAGE))
            self.leaves = self.model.tree.get_leaves()
        else:
            logging.info('Extracting a Black Box Model')
            self.model_id = black_box_models[data]

            # get the black-box model with the real credentials for sanity
            # checks
            try:
                self.model = Model('model/{}'.format(self.model_id),
                                   api=BigML(username='ftramer2',
                                             api_key=BB_KEY))
            except ValueError:
                self.model = Model('public/model/{}'.format(self.model_id),
                                   api=BigML(storage=STORAGE))

            self.connection = BigML()

        TreeExtractor.__init__(self, epsilon, rounding)

    def get_classes(self):
        tree = self.model.tree
        for key, val in util.sort_fields(tree.fields):
            if key == tree.objective_id:
                return [str(x[0]) for x in val['summary']['categories']]

    def get_features(self):
        """
        Parse the BigML tree model to get all the features
        """
        features = []
        tree = self.model.tree
        for key, val in util.sort_fields(tree.fields):
            if key and key != tree.objective_id:
                ftype = str(val['optype'])
                if ftype == "numeric":
                    features.append(ContFeature(str(val['name']), key,
                                                val['summary']['minimum'],
                                                val['summary']['maximum']))
                elif ftype == "categorical":
                    categories = sorted([str(name) for (name, _)
                                         in val['summary']['categories']])
                    features.append(CatFeature(str(val['name']), key,
                                               list(categories)))
                else:
                    raise ValueError("Unknown feature type {}".format(ftype))
        return features

    def make_prediction(self, query):
        if not self.black_box:
            res = self.model.predict(query,
                                     add_confidence=True,
                                     add_distribution=True,
                                     add_path=True,
                                     add_next=True)

            # simulate the "fields" information in the prediction response
            features = get_features_on_prediction_path(res['path'])
            features = features.union(set(query.keys()))
            if res['next']:
                features = features.union(set([str(res['next'])]))

            res_id = LeafID(res['prediction'], res['confidence'],
                            self.rounding, features)
            return res_id
        else:
            logging.info('{}: {}'.format(self.queryCount, query))
            headers = {'content-type': 'application/json'}
            url = 'https://bigml.io/prediction'+self.connection.auth

            payload = {
                "model": "public/model/{}".format(self.model_id),
                "input_data": query
            }

            r = requests.post(url, data=json.dumps(payload), headers=headers)
            print 'request took {} ms'.format(1000*r.elapsed.total_seconds())

            res = r.json()
            fields = [str(f['name']) for (k, f) in res['fields'].iteritems()
                      if k != res['objective_field']]
            res_id = LeafID(res['prediction'].values()[0], res['confidence'],
                            self.rounding, fields)
            logging.info('{}'.format(res_id))

            return res_id

    def get_leaves(self):
        if not self.black_box:
            paths = [parse_path(leaf['path'], self.features)
                     for leaf in self.leaves]

            return [(LeafID(leaf['output'], leaf['confidence'],
                            self.rounding, predicate_names(path)), path)
                    for (leaf, path) in zip(self.leaves, paths)]
        else:
            raise NotImplementedError()


def predict(nodes, x):
    for (id, l) in nodes.iteritems():
        for node in l:
            if pred_match(x, node):
                return id.val

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='a dataset')
    parser.add_argument('--top', action='store_true', dest="top",
                        help='top down algorithm mode')
    parser.add_argument('--incomplete', dest='incomplete',
                        action='store_true', help='allow incomplete queries')
    parser.add_argument('--bb', dest='black_box',
                        action='store_true', help='black box model')
    parser.add_argument('--rounding', type=int, help='rounding digits')
    parser.add_argument('--epsilon', type=float, help='precision', default=1e-3)
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    rounding = args.rounding
    epsilon = args.epsilon
    black_box = args.black_box

    if args.data == "bitcoin":
        epsilon = 1e-4

    incomplete_queries = args.incomplete
    seed = args.seed
    verbose = args.verbose

    if verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    np.random.seed(seed)

    ext = BigMLTreeExtractor(args.data, epsilon=epsilon, rounding=rounding,
                             black_box=black_box)

    start_time = timeit.default_timer()

    if not args.top:
        all_leaves = ext.extract(incomplete_queries=incomplete_queries)
    else:
        all_leaves = ext.extract_top_down()

    end_time = timeit.default_timer()
    budget = ext.queryCount

    print 'found {} leaves ({} unique)'.format(
        sum([len(x) for x in all_leaves.values()]), len(all_leaves))
    print 'Extraction required {} queries'.format(budget)
    print 'Extraction took %.2f seconds' % (end_time - start_time)
    asdf

    leaves_with_paths = ext.get_leaves()
    leaves = [l for (l, _) in leaves_with_paths]
    unique_leaves = set(leaves)

    not_found_count = 0
    for (leaf, path1) in \
            sorted(leaves_with_paths, key=lambda (id, path): (id.val, id.conf)):
        found = False
        for path2 in all_leaves.get(leaf, []):
            if ext.equal_paths(path1, path2):
                found = True
        if not found:
            not_found_count += 1

            logging.info('leaf {} was not found!'.format(leaf))
            logging.info(path1)
            for path2 in all_leaves.get(leaf, []):
                logging.info(path2)
            logging.info('----------------')

    logging.info('SUMMARY:')
    try:
        logging.info('Model has {} classes: {}'.format(len(ext.get_classes()),
                                                       ext.get_classes()))
    except KeyError:
        logging.info('Regression Tree')
    logging.info('Model has {} features'.format(len(ext.get_features())))
    logging.info('Tree has {} leaves ({} unique) - Max depth: {}'.
                 format(len(leaves), len(unique_leaves),
                        max([len(leaf['path']) for leaf in ext.leaves])))

    logging.info('found {} leaves ({} unique)'.
                 format(sum([len(x) for x in all_leaves.values()]),
                        len(all_leaves)))

    logging.info('{} leaves were not found!'.format(not_found_count))

    logging.info('Extraction required {} queries'.format(budget))

    X_test_u = ext.generate_uniform_queries(10000)

    y_true = [str(ext.predict(x).val) for x in X_test_u]
    y_pred = [str(predict(all_leaves, x)) for x in X_test_u]

    count_missed = len([x for x in y_pred if x == 'None'])
    logging.info('Unif Missed: {}'.format(count_missed))

    acc_u = accuracy_score(y_true, y_pred)
    logging.info('L_unif: {:.2e}'.format(1-acc_u))

    target = ext.model.fields[ext.model.objective_id]['name']
    X_test, _ = prepare_data(args.data, target)

    s1 = set(X_test.columns.values)
    s2 = set([f.name for f in ext.features])

    assert s2.issubset(s1)

    X_test = X_test.to_dict('records')[0:10000]

    y_true = [str(ext.predict(x).val) for x in X_test]
    y_pred = [str(predict(all_leaves, x)) for x in X_test]

    count_missed = len([x for x in y_pred if x == 'None'])
    logging.info('Test Missed: {}'.format(count_missed))

    acc = accuracy_score(y_true, y_pred)
    logging.info('L_test: {:.2e}'.format(1-acc))

    method = "top-down" if args.top else "bottom-up"

    print '%s, %s, %d, %d, %.2e, %.2e, %d, %.2f' \
          % (args.data, method, len(leaves), len(unique_leaves),
             1-acc, 1-acc_u, budget, end_time - start_time)

if __name__ == "__main__":
    main()
