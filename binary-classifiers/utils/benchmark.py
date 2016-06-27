__author__ = 'Fan'

from os.path import isfile
from os import rename
import pickle


class Benchmark:
    def __init__(self, name):
        self.name = name

        self.record = {}
        if isfile(name + '.pkl'):
            with open(self.name + '.pkl', 'rb') as infile:
                try:
                    self.record = pickle.load(infile)
                    print 'done loading from %s' % name + '.pkl'
                except:
                    rename(name + '.pkl', name + '.pkl.org')

    def add_kv(self, key, value):
        self.record.update({key: value})

    def add(self, item, x, xname, ys, ynames, yys, yynames):
        assert len(ys) == len(ynames)
        assert len(yys) == len(yynames)

        if item not in self.record:
            self.record[item] = {
                'x': x,
                'xname': xname,
                'ny': len(ys),
                'nyy': len(yys),
            }
            r = self.record[item]
            for i, y in enumerate(ys):
                r['y%d' % i] = [y]
                r['y%dname' % i] = ynames[i]

            for ii, yy in enumerate(yys):
                r['yy%d' % ii] = [yy]
                r['yy%dname' % ii] = yynames[ii]

        else:
            r = self.record[item]
            for i, y in enumerate(ys):
                assert r['y%dname' % i] == ynames[i]
                r['y%d' % i].append(y)

            for ii, yy in enumerate(yys):
                assert r['yy%dname' % ii] == yynames[ii]
                r['yy%d' % ii].append(yy)

    def store(self):
        with open(self.name + '.pkl', 'wb') as out:
            pickle.dump(self.record, out)

