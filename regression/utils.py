import numpy as np
from sklearn.linear_model.logistic import safe_sparse_dot
from sklearn.utils.extmath import squared_norm, log_logistic
from scipy.special import expit
import pandas as pd
import sklearn.datasets
try:
    import matplotlib.pyplot as plt
except:
    plt = None 
    pass
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from scipy.spatial.distance import cdist, pdist, squareform
from PIL import Image
from munkres import Munkres
import os
import errno
import math



SCALE_TYPE = "uniform"
bounds = None


class DummyScaler:
    def fit_transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


def prepare_data(name, onehot=True, labelEncode=True):
    if name == "iris" or name == "irisQB":
        X, y = prepare_iris()
    elif name == "iris_b":
        X, y = prepare_iris(binary=True)
    elif name == "diabetes" or name == "diabetesQB":
        X, y = prepare_diabetes()
    elif name == "mushrooms":
        X, y = prepare_mushrooms()
    elif name == "cancer" or name == "cancerQB":
        X, y = prepare_cancer()
    elif name == "adult":
        X, y = prepare_adult(target="race", onehot=onehot)
    elif name == "adult_b" or name == "adult10QB" or name == "adultQB":
        X, y = prepare_adult(target="income", onehot=onehot)
    elif name == "steak":
        X, y = prepare_steak(onehot=onehot)
    elif name == "gss":
        X, y = prepare_gss(onehot=onehot)
    elif name == "moons":
        X, y = prepare_moons()
    elif name == "circles":
        X, y = prepare_circles()
    elif name == "circlesQB":
        X, y = prepare_circlesQB()
    elif name == "blobs":
        X, y = prepare_blobs()
    elif name == "class5":
        X, y = prepare_classification(num_classes=5)
    elif name == "digits" or name == "digits2":
        X, y = prepare_digits()
    elif name == "digits40":
        X, y = prepare_digits()
        X, y = X.values[0:40, :], y.values[0:40]
    elif name == "digits_all":
        X, y = prepare_digits_all()
    elif name == "faces":
        return prepare_faces()
    elif name == "att_faces":
        return prepare_att_faces()
    else:
        raise ValueError('Unknown dataset %s', name)

    if SCALE_TYPE in ["uniform", "uniform_int", "norm"]:
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        scaler = DummyScaler()

    if not onehot:
        X = X.values
        for i in range(X.shape[1]):
            try:
                X[:, i] = scaler.fit_transform(X[:, i])
            except ValueError:
                if labelEncode:
                    X[:, i] = LabelEncoder().fit_transform(X[:, i])

        X = pd.DataFrame(X)
    else:
        X = pd.DataFrame(scaler.fit_transform(X))
    y = pd.Series(LabelEncoder().fit_transform(y))

    if SCALE_TYPE == "data":
        global bounds
        bounds = [X.min(axis=0), X.max(axis=0)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

    #X_train, y_train = X, y
    #X_test, y_test = X.copy(), y.copy()

    return X_train, y_train, X, y, scaler


def prepare_att_faces():
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk('../data/att_faces'):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                im = Image.open(os.path.join(subject_path, filename))
                im = im.convert("L")
                X.append(np.asarray(im, dtype=np.uint8).flatten())
                y.append(c)
            c = c + 1
    return None, None, np.array(X), np.array(y), None 

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
    y_train = pd.Series(np.concatenate(y_train))
    y_test = pd.Series(np.concatenate(y_test))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))

    return X_train, y_train, X_test, y_test, scaler


def prepare_adult(target="income", onehot=True):
    data = pd.read_csv('../data/adult.csv', sep=r'\s*,\s*', engine='python')

    cols = list(data.columns.values)
    cols.remove(target)

    X = data[cols]
    y = data[target]

    if onehot:
        X = pd.get_dummies(X)

    return X, y


def prepare_steak(onehot=True):
    data = pd.read_csv('../data/steak.csv').dropna()
    target = "How do you like your steak prepared?"

    del data['RespondentID']

    X = data[list(set(data.columns) - set([target]))]
    y = data[target]

    if onehot:
        X = pd.get_dummies(X)

    return X, y


def prepare_gss(onehot=True):
    data = pd.read_csv('../data/GSShappiness.csv')

    del data['year']
    del data['id']

    data = data.dropna()
    target = "Happiness level"

    X = data[list(set(data.columns) - set([target]))]
    y = data[target]

    if onehot:
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


def prepare_blobs():
    X, y = sklearn.datasets.make_blobs(5000, cluster_std=0.5)
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


def prepare_circlesQB():
    X, y = sklearn.datasets.load_svmlight_file(
        '../binary-classifiers/targets/circle/test.scale')
    return pd.DataFrame(X.toarray()), pd.Series(y)


def prepare_circles():
    X, y = sklearn.datasets.make_circles(5000, factor=.5, noise=.05)
    return X, y


def prepare_moons():
    X, y = sklearn.datasets.make_moons(5000, noise=0.25)
    return X, y


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


def prepare_classification(num_classes=5):
    X, y = sklearn.datasets.make_classification(n_samples=1000,
                                                n_classes=num_classes,
                                                n_informative=4)

    return X, y


def gen_query_set(n, test_size=100000, dtype=SCALE_TYPE):
    """
    Produce a random vector of size n with values in the range [low,high)
    """

    if dtype == "uniform":
        return np.random.uniform(-1, 1, size=(test_size, n))
    elif dtype == "uniform_int":
        return 2 * np.random.randint(2, size=(test_size, n)) - 1
    elif dtype == "norm":
        return np.random.randn(test_size, n)
    elif dtype == "data":
        min_x, max_x = bounds
        data = np.zeros((test_size, n))
        for i in range(n):
            data[:, i] = np.random.uniform(min_x[i], max_x[i], size=test_size)
        return data
    else:
        raise ValueError("Unknown data type")


def stat_distance(p, q):
    assert p.shape == q.shape

    tot = 0.5 * np.abs((p-q)).sum()

    return tot/len(p)


def plot_decision_boundary(pred_func, X, y, bounds, filename=None):
    if plt is None:
        return

    fig = plt.figure()
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
    return fig


def compare_boundaries(pred_func1, pred_func2, bounds, filename=None):
    if plt is None:
        return
    # Set min and max values and give it some padding
    x_min, x_max, y_min, y_max = bounds
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
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


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


def line_search_oracle(n, budget, predict_func, query_gen, eps=1e-1):
    X_init = query_gen(n, 1)
    Y = predict_func(X_init)

    tot_budget = budget
    budget -= 1

    step = (budget+3)/4

    while query_count(X_init, Y, eps) <= budget:
        x = query_gen(n, step)
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


def approx_fprime_helper(xk, f, epsilon, args=(), f0=None):
    """
    See ``approx_fprime``.  An optional initial function value arg is added.
    """
    if f0 is None:
        f0 = f(*((xk,) + args))
    grad = np.zeros(xk.shape, float)
    ei = np.zeros(xk.shape, float)

    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        grad[k] = (f(*((xk + d,) + args)) - f0) / d[k]
        ei[k] = 0.0
    return grad


def temp_log_loss(w, X, Y, alpha):
    n_classes = Y.shape[1]
    w = w.reshape(n_classes, -1)
    intercept = w[:, -1]
    w = w[:, :-1]
    z = safe_sparse_dot(X, w.T) + intercept

    denom = expit(z)
    #print denom
    #print denom.sum()
    denom = denom.sum(axis=1).reshape((denom.shape[0], -1))
    #print denom
    p = log_logistic(z)

    loss = - (Y * p).sum()
    loss += np.log(denom).sum()
    loss += 0.5 * alpha * squared_norm(w)

    return loss


def min_l1_dist(m1, m2):
    assert len(m1) == len(m2)

    # pairwise l1 distances
    dist = cdist(m1, m2, 'minkowski', 1)

    m = Munkres()
    matching = m.compute(dist.copy())

    total = 0.0
    for row, column in matching:
        value = dist[row][column]
        total += value

    return total, matching


def create_dir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


