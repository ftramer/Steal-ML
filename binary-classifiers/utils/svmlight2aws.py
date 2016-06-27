from sklearn.datasets import load_svmlight_file
import csv
import argparse
import os
import numpy as np

BASE_DIR = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-n', '--num_of_features', required=True, type=int)
args = vars(parser.parse_args())

n_features = args['num_of_features']
input = args['input']

inputfile = os.path.join(BASE_DIR, input)

x, y = load_svmlight_file(inputfile)
x = x.todense().tolist()

outputfile = inputfile + '.csv'

print 'Written to %s' % outputfile

with open(outputfile, 'w') as fp:
    ww = csv.writer(fp)
    for yy, xx in zip(y, x):
        # if yy != 1:
        #     yy = 0
        a = [int(yy)]
        a.extend(xx)
        ww.writerow(a, )