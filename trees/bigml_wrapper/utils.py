#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from predicate import *
import pandas as pd


def parse_pred(p, features):
    """
    Parse a predicate from a string description
    """
    parser = re.search('\((.*) \(f (.*)\) (.*)\)', p)
    op = parser.group(1)
    field_key = parser.group(2)
    feature = [f for f in features if f.key == field_key][0]
    value = parser.group(3)

    if op == '<=':
        pred = LTE(feature, float(value), None)
    elif op == '>':
        pred = GT(feature, float(value), None)
    elif op == '=':
        pred = CAT(feature, [value], None)
    elif op == '!=':
        pred = CAT(feature, list(set(feature.vals)-set([value])), None)
    else:
        raise ValueError('Unknown OP {}'.format(p))

    return pred


def parse_path(path, features):
    """
    Parse a path of predicates from a string description
    """
    predicates = [parse_pred(p, features) for p in path]
    return predicates


def get_features_on_prediction_path(path):
    """
    Get the names of features on a path
    """
    features = set()
    for pred in path:
        feature_name = re.split(" <= | > | != | = ", pred)[0]
        features.add(feature_name)

    return features


def prepare_data(model, target=None):
    if model == "steak":
        return prepare_steak()
    elif model == "movies":
        return load_data("gss", target)
    else:
        return load_data(model, target)


def prepare_steak():
    data = pd.read_csv('../data/steak.csv').dropna()
    target = "How do you like your steak prepared?"

    del data['RespondentID']
    del data['Do you eat steak?']

    X = data[list(set(data.columns) - set([target]))]
    y = data[target]

    return X, y


def load_data(model, target):
    data = pd.read_csv('../data/bigml_datasets/{}.csv'.format(model)).dropna()
    X = data[list(set(data.columns) - set([target]))]
    y = data[target]

    return X, y


def prepare_movies():
    data = pd.read_csv('../data/GSShappiness.csv')

    del data['year']
    del data['id']

    data = data.dropna()
    target = "watched x-rated movies in the last year"

    X = data[list(set(data.columns) - set([target]))]
    X.columns = [x[0].upper() + x[1:] for x in X.columns]

    X = X.replace({'Age': {'89 OR OLDER': '89'},
                   'Children': {'EIGHT OR MORE': '8'},
                   'Work status': {'UNEMPL, LAID OFF': 'Unemployed, laid off'}
                   })

    X = X.applymap(lambda x: x[0] + x[1:].lower())

    X[['Age', 'Children']] = X[['Age', 'Children']].astype(float)

    y = data[target]

    return X, y
