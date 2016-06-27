__author__ = 'Fan'

import os

from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score

from algorithms.awsOnline import AWSOnline

val_name = map(lambda x: 'v%d' % x, range(1, 9))
print val_name
test = AWSOnline('ml-lzYmNFzPh2N', 1, 0, 8, val_name, 'uniform', .1)

Xt, Yt = load_svmlight_file(os.getenv('HOME') + '/Dropbox/Projects/SVM/dataset/cod-rna/cod-rna.test.scaled',
                            n_features=8)
Xt = Xt.todense().tolist()

import random

z = zip(Xt, Yt)
random.shuffle(z)
Xt, Yt = zip(*z)

start = 0
cutoff = 10

Xt = Xt[start:start + cutoff]

yy = [test.query(x) for x in Xt]
print 'True:', Yt[start:start + cutoff]
print 'Predict:', yy
print accuracy_score(Yt[start:start + cutoff], yy)
