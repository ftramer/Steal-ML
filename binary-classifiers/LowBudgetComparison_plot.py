__author__ = 'fan'

import pickle

import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt

from utils.benchmark import Benchmark
from utils.contour import drawy_yy

f = '/Users/fan/Desktop/diabetes.aa-lb'
b = Benchmark(f)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)

baseline = b.record.pop('baseline')
item = 'retrain in X with grid (low budget)'

v = b.record[item]
x = v['x']
xname = v['xname']

# v['y%d' % i] is a list of arrays
# np.average(np.array(v['y%d' % i]), axis=0) returns average over all arrays
# v = [[1,2], [3,4]] return [2, 3]
ny = v['ny']
nyy = v['nyy']
ys_averaged = [np.average(np.array(v['y%d' % i]), axis=0).tolist() for i in range(0, v['ny'])]
ys_err = [(1.96/sqrt(ny)*np.std(np.array(v['y%d' % i]), axis=0)).tolist() for i in range(0, v['ny'])]
yys_averaged = [np.average(np.array(v['yy%d' % i]), axis=0).tolist() for i in range(0, v['nyy'])]
yys_err = [(1.96/sqrt(ny)*np.std(np.array(v['yy%d' % i]), axis=0)).tolist() for i in range(0, v['nyy'])]
ynames = [v['y%dname' % i] for i in range(0, v['ny'])]
yynames = [v['yy%dname' % i] for i in range(0, v['nyy'])]

##############################
# retrain in F with diff. dims
##############################
pklfile = '/Users/fan/Desktop/diabetes.aa-lb-grid.pkl'
f = open(pklfile, 'rb')
bn_dim, bn_budget, bf_train, bf_test, bn_q = pickle.load(f)
f.close()

data = np.array([bn_dim, bn_budget, bf_test]).T

base = len(x)*[baseline]

ys = [(ys_averaged[1], 'o-', 'retrain in $\mathcal{X}$')]
yys = ()

dims = list(set(bn_dim))
dims.sort()
for dim in dims:
    if dim < 1000:
        continue
    bud = data[data[:, 0] == dim][:, 1]
    score = data[data[:, 0] == dim][:, 2]
    ys.append((score.tolist(), '^-', 'retrain in $\mathcal{F}$, RBF dim. = $%d$' % dim))

drawy_yy(ax, x, base, ys, yys, xname, 'score', 'num of queries', 'Low budget comparison')

pklfile.replace('.', '-')
plt.savefig(pklfile + '.pdf')