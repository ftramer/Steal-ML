import boto3
import numpy as np
from copy import copy
from scipy.special import logit
from models import models
import decimal
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import re
from math import exp

client = boto3.client('machinelearning')

model = models['adult10QB']
model_id = model['id']

aws_model = client.get_ml_model(
    MLModelId=model_id,
    Verbose=True
)
schema = aws_model['Schema']
target_reg = '"targetAttributeName":"(.*?)"'
target = re.findall(target_reg, schema)[0]
feature_reg = '"attributeName":"(.*?)","attributeType":"(.*?)"'
feature_info = re.findall(feature_reg, schema)
val_name = [f for (f, t) in feature_info if f != target]

def predict(query):
    response = client.predict(
        MLModelId=model_id,
        Record={k:str(v) for (k,v) in query.iteritems()},
        PredictEndpoint='https://realtime.machinelearning.us-east-1.amazonaws.com'
    )
    
    p = response['Prediction']['predictedScores'].values()
    return set(p)

def make_query(query, feature_name, val):
    """
    Modify a given query by switching the value of one feature
    """
    new_query = copy(query)
    new_query[feature_name] = val
    return new_query

def eps_round(x, epsilon):
    """
    Round a floating point value to the nearest multiple of eps
    """
    return round(x / epsilon) * epsilon


def line_search(query, feature, min_val, max_val, min_p, max_p, epsilon):
    """
    Perform a line search on a continuous feature to find all splitting
    thresholds
    """
    prec = -decimal.Decimal(str(epsilon)).as_tuple().exponent

    def int_line_search(query, feature, min_mult, max_mult, min_p, max_p):
        if min_p == max_p:
            split = (min_mult + max_mult)/2
            split_query = make_query(query, feature, round(split * epsilon, prec))
            split_p = predict(split_query)

            if split_p == min_p:
                return []

        # if we've narrowed down the split to less than epsilon,
        # call it a day
        if (max_mult - min_mult) <= 1:
            threshold = round(min_mult * epsilon, prec)
            print 'ending line search with {} ({} - {}) for initial query {}'.format(threshold, min_mult, max_mult, query)
            return [threshold]

        # split at half way and check recursively on both sides
        split = (min_mult + max_mult)/2
        split_query = make_query(query, feature, round(split * epsilon, prec))
        split_p = predict(split_query)

        print '\tval {} got {}'. format(round(split*epsilon, prec), split_p)
        return int_line_search(query, feature, split, max_mult, split_p, max_p) + \
               int_line_search(query, feature, min_mult, split, min_p, split_p)

    max_mult = int(round(max_val / epsilon))
    min_mult = int(round(min_val / epsilon))
    return int_line_search(query, feature, min_mult, max_mult, min_p, max_p)

q0 = np.zeros(len(val_name))
q0 = dict(zip(val_name, map(str, q0)))

p = predict(q0)
print '{} got {}'. format(None, p)

for val in [-1, -1.01, -1, -0.999, 0, 1]:
    temp = make_query(q0, 'capital-loss', val)
    p = predict(temp)
    print '{} got {}'. format(val, p)
