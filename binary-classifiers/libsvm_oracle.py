__author__ = 'Fan'

import sys

sys.path.append('./libsvm-3.20/python')

from svmutil import *

working_dir = os.path.join(os.getenv('HOME'), 'Dropbox/Projects/SVM/cod-rna/')
y, x = svm_read_problem(working_dir + 'cod-rna')

# m = svm_train(y, x, '-h 0')
# svm_save_model(working_dir + 'cod-ran.model', m)
m = svm_load_model(working_dir + 'cod-ran.model')

p_label, p_acc, p_val = svm_predict(y, x, m)
# print p_label
print p_acc
# print p_val
