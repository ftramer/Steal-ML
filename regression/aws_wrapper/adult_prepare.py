import pandas as pd
import numpy as np
from models import models
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


data = pd.read_csv('../data/aws_adult.csv')

X = data[data.columns[0:-1]]
cols = X.columns

y = data[data.columns[-1]]

scaler = MinMaxScaler(feature_range=(-1, 1))
X = X.values
for i in range(X.shape[1]):
    try:
        X[:, i] = scaler.fit_transform(X[:, i])
    except ValueError:
        X[:, i] = LabelEncoder().fit_transform(X[:, i])
X = pd.DataFrame(X)
X.columns = cols

X = X.iloc[0:int(0.7*len(X))]

num_bins = 10

for f in ['age', 'fnlwgt', 'education-num',
          'capital-gain', 'capital-loss', 'hours-per-week']:
    bins = []
    for i in range(1, num_bins+1):
        new = round(np.percentile(X[f], (100.0/num_bins)*i), 3)
        if new not in bins:
            bins.append(new)

    print '{}: {} ({})'.format(f, bins, len(bins))
