import utils
import pandas
from sklearn.preprocessing import MinMaxScaler

X = pandas.read_csv('../data/adult.csv')

scaler = MinMaxScaler(feature_range=(-1, 1))
X = X.values
for i in range(X.shape[1]):
    try:
        X[:, i] = scaler.fit_transform(X[:, i])
    except ValueError:
        pass

y = []
for i in range(X.shape[0]):
    a = X[i, 14]
    if a.strip().startswith('<'):
        X[i, 14] = 0
        y.append(0)
    else:
        X[i, 14] = 1
        y.append(1)


import numpy
print numpy.bincount(y)

X = pandas.DataFrame(X)
X.to_csv('adult.csv', index=False)
