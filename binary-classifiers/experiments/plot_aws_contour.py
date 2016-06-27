import sys
sys.path.append('..')
from utils.contour import plotcontour
from algorithms.awsOnline import AWSOnline
from sklearn import datasets

meta = {
    'breast-cancer': {
        'val_name': ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'],
        'model_id': 'ml-lkYRYeldcrH'
    },
    'circle' : {
        'val_name': ['x1', 'x2'],
        'model_id': 'ml-i0GeYZaGQ3f'
    },
    'fourclass' : {
        'val_name': ['x1', 'x2'],
        'model_id': 'ml-Je6DdX8c57P'
    },
    'diabetes': {
        'val_name': ['x' + str(i) for i in range(1, 10 + 1)],
        'model_id': 'ml-UGnrMStrX2o'
    },
    'circle-wo-qb': {
        'val_name': ['x1', 'x2'],
        'model_id': 'ml-utTb9cZRabu'
    }
}

dataset = 'circle-wo-qb'
n_features = 2
ex = AWSOnline(meta[dataset]['model_id'], 1, 0, n_features, meta[dataset]['val_name'], ftype='uniform', error=.1)
X, y = datasets.make_circles(n_samples=800, noise=0.07, factor=0.4)
plotcontour(ex.batch_predict, X, y)
