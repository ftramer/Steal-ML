import sklearn.datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np


digits = sklearn.datasets.load_digits()
X = digits.data
y = digits.target

scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)

len_train = 0.7 * len(X)
X_train = X[0:len_train]
X_test = X[len_train:]
y_train = y[0:len_train]
y_test = y[len_train:]

gamma = 2**(-3)

reg = LogisticRegression(C=1e6)
reg.fit(rbf_kernel(X_train, X_train, gamma), y_train)
#reg.fit(X_train, y_train)

y_pred = reg.predict(rbf_kernel(X_test, X_train, gamma))
#y_pred = reg.predict(X_test)

print classification_report(y_test, y_pred)