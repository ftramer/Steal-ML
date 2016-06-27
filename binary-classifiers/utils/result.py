__author__ = 'Fan'

import numpy as np


class Result(object):
    def __init__(self, name, aws=False):
        self.name = name

        self.Q_by_U = (1, 2, 5, 8, 10, 20, 50, 80, 100,)
        if aws:
            # self.Q_by_U = (10, 20, 50, 100, 150, 200, 500,)
            self.Q_by_U = (50, )

        self.index  = range(0, len(self.Q_by_U))

        assert len(self.Q_by_U) == len(self.index)

        self.L_unif = {}
        self.L_test = {}
        self.nquery = {}

        for i in self.index:
            self.L_unif[i] = []
            self.L_test[i] = []
            self.nquery[i] = []

    def __str__(self):
        s = '# ' + self.name + '\n'
        s += '# Q_by_U, q, L_unif_bar, L_test_bar, L_unif_std, L_test_std\n'
        for i in self.index:
            s += '%d, %d, %f, %f, %f, %f\n' % (
                self.Q_by_U[i],
                np.average(self.nquery[i]),
                np.average(self.L_unif[i]),
                np.average(self.L_test[i]),
                np.std(self.L_unif[i]),
                np.std(self.L_test[i]),)

        return s



if __name__ == '__main__':
    r = Result('test')
    for q in r.index:
        r.L_unif[q].append(q)
        r.L_test[q].append(q)
        r.nquery[q].append(q)

    print r