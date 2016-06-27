import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from munkres import Munkres
import sklearn.datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split

try:
    import matplotlib.pyplot as plt
except:
    plt = None
    pass
import os
import errno
import math


SCALE_TYPE = "uniform"
#SCALE_TYPE = "norm"


def gen_query_set(n, test_size, dtype=SCALE_TYPE):
    """
    Produce a random vector of size n with values in the range [low,high)
    """
    if dtype == "uniform":
        return np.random.uniform(-1, 1, size=(test_size, n))
    elif dtype == "uniform_int":
        return 2 * np.random.randint(2, size=(test_size, n)) - 1
    elif dtype == "norm":
        return np.random.randn(test_size, n)
    else:
        raise ValueError("Unknown data type")


def prepare_data(name):
    if name == "circles":
        X, y = prepare_circles()
    elif name == "blobs":
        X, y = prepare_blobs()
    elif name == "moons":
        X, y = prepare_moons()
    elif name == "adult":
        X, y = prepare_adult(target="race")
    elif name == "adult_b":
        X, y = prepare_adult(target="income")
    elif name == "class5":
        X, y = prepare_classification()
    elif name == "digits":
        X, y = prepare_digits()
    elif name == "digits_all":
        X, y = prepare_digits_all()
    elif name == "faces":
        return prepare_faces()
    elif name == "steak":
        X, y = prepare_steak()
    elif name == "gss":
        X, y = prepare_gss()
    elif name == "iris":
        X, y = prepare_iris()
    elif name == "diabetes":
        X, y = prepare_diabetes()
    elif name == "mushrooms":
        X, y = prepare_mushrooms()
    elif name == "cancer":
        X, y = prepare_cancer()
    else:
        raise ValueError('Unknown dataset %s' % name)

    if SCALE_TYPE == "uniform":
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        scaler = StandardScaler()

    X = pd.DataFrame(scaler.fit_transform(X))
    y = pd.Series(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

    return X_train, y_train, X, y, scaler


def prepare_adult(target="income"):
    data = pd.read_csv('../data/adult.csv')

    X = data[list(set(data.columns) - set([target]))]
    y = data[target]

    X = pd.get_dummies(X)

    return X, y


def prepare_iris(binary=False):
    data = pd.read_csv('../data/iris.csv')

    if binary:
        data = data[data[' class'] != 'Iris-setosa']

    data = data.iloc[np.random.permutation(np.arange(len(data)))]

    X = data[data.columns[:-1]]
    y = data[data.columns[-1]]

    return X, y


def prepare_classification(num_classes=5):
    X, y = sklearn.datasets.make_classification(n_samples=1000,
                                                n_classes=num_classes,
                                                n_informative=4)
    return X, y


def prepare_diabetes():
    X, y = sklearn.datasets.load_svmlight_file(
        '../binary-classifiers/targets/diabetes/diabetes')
    return X.toarray(), y


def prepare_mushrooms():
    X, y = sklearn.datasets.load_svmlight_file(
        '../binary-classifiers/targets/mushrooms/mushrooms')
    return X.toarray(), y


def prepare_cancer():
    X, y = sklearn.datasets.load_svmlight_file(
        '../binary-classifiers/targets/breast-cancer/train.scale')
    return X.toarray(), y


def prepare_moons():
    X, y = sklearn.datasets.make_moons(5000, noise=0.25)
    return X, y


def prepare_blobs():
    X, y = sklearn.datasets.make_blobs(5000, cluster_std=0.5)
    return X, y


def prepare_circles():
    X, y = sklearn.datasets.make_circles(5000, factor=.5, noise=.05)
    return X, y


def prepare_faces():
    data = sklearn.datasets.fetch_olivetti_faces('../data', shuffle=False)
    X = data.data
    y = data.target

    X = np.split(X, 40)
    y = np.split(y, 40)

    X_train = [x[0:7, :] for x in X]
    X_test = [x[7:, :] for x in X]
    y_train = [a[0:7] for a in y]
    y_test = [a[7:] for a in y]
    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    y_train = np.concatenate(y_train)
    y_test = np.concatenate(y_test)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, scaler


def prepare_digits():
    digits = sklearn.datasets.load_digits()
    X = pd.DataFrame(digits.data)
    y = pd.Series(digits.target)
    return X, y


def prepare_digits_all():
    data = sklearn.datasets.fetch_mldata('MNIST original', data_home='../data')
    X = data.data.astype(float)
    y = np.asarray(data.target, dtype=int)
    return X, y


def prepare_steak():
    data = pd.read_csv('../data/steak.csv').dropna()
    target = "How do you like your steak prepared?"

    del data['RespondentID']

    X = data[list(set(data.columns) - set([target]))]
    y = data[target]

    X = pd.get_dummies(X)

    return X, y


def prepare_gss():
    data = pd.read_csv('../data/GSShappiness.csv')

    del data['year']
    del data['id']

    data = data.dropna()
    target = "Happiness level"

    X = data[list(set(data.columns) - set([target]))]
    y = data[target]

    X = pd.get_dummies(X)

    return X, y


def stat_distance(p, q):
    assert p.shape == q.shape

    tot = 0.5 * np.abs((p-q)).sum()

    return tot/len(p)


def min_l1_dist(m1, m2):
    assert len(m1) == len(m2)

    # pairwise l1 distances
    dist1 = cdist(m1, m2, 'minkowski', 1)
    dist2 = cdist(m1, -m2, 'minkowski', 1)

    neg = zip(*np.where(dist1 > dist2))

    dist = np.minimum(dist1, dist2)

    m = Munkres()
    matching = m.compute(dist.copy())

    total = 0.0
    for row, column in matching:
        value = dist[row][column]
        total += value

    return total, matching, neg


def plot_decision_boundary(pred_func, X, y, bounds, filename=None):
    if plt is None:
        return
    plt.figure()
    h = 0.01
    # Generate a grid of points with distance h between them
    x_min, x_max, y_min, y_max = bounds
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def compare_decision_boundary(pred_func1, pred_func2, X, filename):
    if plt is None:
        return
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z1 = pred_func1(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z1.reshape(xx.shape)

    plt.figure()
    # Plot the contour and training examples
    plt.contour(xx, yy, Z1, cmap=plt.cm.Reds)

    Z2 = pred_func2(np.c_[xx.ravel(), yy.ravel()])
    Z2 = Z2.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contour(xx, yy, Z2, cmap=plt.cm.Blues)
    plt.savefig(filename)
    plt.close()


def line_search(X, Y, num_samples, predict_func, eps=1e-1):
    # random points
    idx1 = np.random.choice(range(len(X)), num_samples)

    # random points from a different class
    idx2 = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        idx2[i] = np.random.choice(np.where(Y != Y[idx1[i]])[0])

    return _line_search(X, Y, idx1, idx2, predict_func, eps)


def _line_search(X, Y, idx1, idx2, predict_func, eps, append=False):
    v1 = X[idx1, :]
    y1 = Y[idx1]
    v2 = X[idx2, :]
    y2 = Y[idx2]

    assert np.all(y1 != y2)

    if append:
        samples = X

    # process all points in parallel
    while np.any(np.sum((v1 - v2)**2, axis=-1)**(1./2) > eps):
        # find all mid points
        mid = 0.5 * (v1 + v2)

        # query the class on the current model
        y_mid = predict_func(mid)

        # change either v1 or v2 depending on the value of y_mid
        index1 = np.where(y_mid != y1)[0]
        index2 = np.where(y_mid == y1)[0]

        if len(index1):
            v2[index1, :] = mid[index1, :]
        if len(index2):
            v1[index2, :] = mid[index2, :]

        if append:
            samples = np.vstack((samples, mid))

    if append:
        return samples
    else:
        return np.vstack((v1, v2))


def all_pairs(Y):
    classes = pd.Series(Y).unique().tolist()
    return [(i, j)
            for i in range(len(Y))              # go over all points
            for c in classes                    # and all other classes
            if c != Y[i]
            for j in np.where(Y == c)[0][0:1]   # and build a pair
            if i > j]


def query_count(X, Y, eps):
    dist = squareform(pdist(X, 'euclidean'))
    tot = 0

    for (i, j) in all_pairs(Y):
        if dist[i][j] > eps:
            tot += math.ceil(np.log2(dist[i][j]/eps))

    return tot


def line_search_oracle(n, budget, predict_func, eps=1e-1):
    X_init = gen_query_set(n, 1)
    Y = predict_func(X_init)

    tot_budget = budget
    budget -= 1

    step = (budget+3)/4

    while query_count(X_init, Y, eps) <= budget:
        x = gen_query_set(n, step)
        y = predict_func(x)
        X_init = np.vstack((X_init, x))
        Y = np.hstack((Y, y))
        budget -= step

    if budget <= 0:
        assert len(X_init) >= tot_budget
        return X_init[0:tot_budget]

    Y = Y.flatten()
    idx1, idx2 = zip(*all_pairs(Y))
    idx1 = list(idx1)
    idx2 = list(idx2)
    samples = _line_search(X_init, Y, idx1, idx2, predict_func, eps,
                           append=True)

    assert len(samples) >= tot_budget

    return samples[0:tot_budget]


def create_dir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
