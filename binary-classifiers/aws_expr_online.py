__author__ = 'Fan'

import numpy as np

from algorithms.awsOnline import AWSOnline
from algorithms.OnlineBase import FeatureSpec

for i in range(0, 1):
    val_name = map(lambda x: 'v%d' % x, range(1, 11))
    print val_name

    test = AWSOnline('ml-lkYRYeldcrH', 1, 0, 10, val_name, 'uniform', .1)
    # X, Y = test.collect_pts(2, -1)

    spec = FeatureSpec('norm', (-1, 1), (-.85, -.85, -.20, -.51, -.49, -.58, -.48, -.38, -.44, -.55))
    X, Y = test.collect_universe(1000, spec=spec)
    print np.bincount(Y)
