import abc
import numpy as np
import pandas as pd
from sklearn.linear_model.logistic import _multinomial_loss, \
    _multinomial_loss_grad, safe_sparse_dot

from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.utils.extmath import squared_norm, log_logistic
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.dummy import DummyClassifier
import utils
import warnings
import timeit
import sys
import decimal


def softmax(X, copy=True):
    """
    Calculate the softmax function.
    The softmax function is calculated by
    np.exp(X) / np.sum(np.exp(X), axis=1)
    This will cause overflow when large values are exponentiated.
    Hence the largest value in each row is subtracted from each data
    point to prevent this.
    Parameters
    ----------
    X: array-like, shape (M, N)
        Argument to the logistic function
    copy: bool, optional
        Copy X or not.
    Returns
    -------
    out: array, shape (M, N)
        Softmax function evaluated at every point in x
    """
    if copy:
        X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X


def predict_probas(X, w, intercept, multinomial=True):
    """
    Predict probabilities for each class using either a multinomial or a
    one-vs-rest approach
    """

    #print X.shape
    #print w.shape
    #print intercept.shape

    p = safe_sparse_dot(X, w.T, dense_output=True) + intercept

    if multinomial:
        return softmax(p, copy=False)
    else:
        p = p.ravel() if p.shape[1] == 1 else p

        p *= -1
        np.exp(p, p)
        p += 1
        np.reciprocal(p, p)

        if p.ndim == 1:
            return np.vstack([1 - p, p]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            p /= p.sum(axis=1).reshape((p.shape[0], -1))
            return p


def score_function(X, w, intercept):
    """
    Score function to predict classes
    """
    scores = safe_sparse_dot(X, w.T, dense_output=True) + intercept
    return scores.ravel() if scores.shape[1] == 1 else scores


def predict_classes(X, w, intercept, classes):
    """
    Predict class labels for samples in X.
    """
    scores = score_function(X, w, intercept)

    if len(scores.shape) == 1:
        indices = (scores > 0).astype(np.int)
    else:
        indices = scores.argmax(axis=1)

    return classes[indices]


def multinomial_loss(w, X, Y, alpha):
    """
    Wrapper for the multinomial loss function used in scikit
    """
    weights = np.ones((len(X),))
    return _multinomial_loss(w, X, Y, alpha, weights)[0]


def multnomial_grad(w, X, Y, alpha):
    """
    Wrapper for the multinomial gradient function used in scikit
    """
    weights = np.ones((len(X),))
    return _multinomial_loss_grad(w, X, Y, alpha, weights)[1]


def logistic_loss(w, X, Y, alpha):
    """
    Implementation of the logistic loss function when Y is a probability
    distribution.

    loss = -SUM_i SUM_k y_ik * log(P[yi == k]) + alpha * ||w||^2
    """
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    intercept = 0

    if n_classes > 2:
        fit_intercept = w.size == (n_classes * (n_features + 1))
        w = w.reshape(n_classes, -1)
        if fit_intercept:
            intercept = w[:, -1]
            w = w[:, :-1]
    else:
        fit_intercept = w.size == (n_features + 1)
        if fit_intercept:
            intercept = w[-1]
            w = w[:-1]

    z = safe_sparse_dot(X, w.T) + intercept

    if n_classes == 2:
        # in the binary case, simply compute the logistic function
        p = np.vstack([log_logistic(-z), log_logistic(z)]).T
    else:
        # compute the logistic function for each class and normalize
        denom = expit(z)
        denom = denom.sum(axis=1).reshape((denom.shape[0], -1))
        p = log_logistic(z)
        loss = - (Y * p).sum()
        loss += np.log(denom).sum()  # Y.sum() = 1
        loss += 0.5 * alpha * squared_norm(w)
        return loss

    loss = - (Y * p).sum() + 0.5 * alpha * squared_norm(w)
    return loss


def logistic_grad_bin(w, X, Y, alpha):
    """
    Implementation of the logistic loss gradient when Y is a binary probability
    distribution.
    """
    grad = np.empty_like(w)
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = w.size == (n_features + 1)

    if fit_intercept:
        intercept = w[-1]
        w = w[:-1]
    else:
        intercept = 0

    z = safe_sparse_dot(X, w.T) + intercept

    _, n_features = X.shape
    z0 = - (Y[:, 1] + (expit(-z) - 1))

    grad[:n_features] = safe_sparse_dot(X.T, z0) + alpha * w

    if fit_intercept:
        grad[-1] = z0.sum()

    return grad.flatten()
    
    
def logistic_grad(w, X, Y, alpha):
    """
    Implementation of the logistic loss gradient when Y is a multi-ary
    probability distribution.
    """

    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = w.size == (n_classes * (n_features + 1))
    grad = np.zeros((n_classes, n_features + int(fit_intercept)))

    w = w.reshape(n_classes, -1)

    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0

    z = safe_sparse_dot(X, w.T) + intercept

    # normalization factor
    denom = expit(z)
    denom = denom.sum(axis=1).reshape((denom.shape[0], -1))

    #
    # d/dwj log(denom)
    #       = 1/denom * d/dw expit(wj * x + b)
    #       = 1/denom * expit(wj * x + b) * expit(-(wj * x + b)) * x
    #
    # d/dwj -Y * log_logistic(z)
    #       = -Y * expit(-(wj * x + b)) * x
    #
    z0 = (np.reciprocal(denom) * expit(z) - Y) * expit(-z)
    
    grad[:, :n_features] = safe_sparse_dot(z0.T, X)
    grad[:, :n_features] += alpha * w

    if fit_intercept:
        grad[:, -1] = z0.sum(axis=0)

    return grad.ravel()


class RegressionExtractor(object):
    """
    Extract coefficients of a logistic regression model.
    """

    def __init__(self):
        self.classes = self.get_classes()
        self.X_train = None

    @abc.abstractmethod
    def num_features(self):
        return

    @abc.abstractmethod
    def get_classes(self):
        return

    def gen_query_set(self, n, test_size, force_input_space=True):
        return utils.gen_query_set(n, test_size)

    def run_opti(self, loss, grad, X, Y, w_dim):
        """
        Wrapper for the optimization procedure.
        Try different regularizers, optimization strategies and random starting
        points until we achieve an overwhelming accuracy on the training set
        """
        k = Y.shape[1]

        best_w = None
        best_int = None
        best_acc = 0

        maxiter = 1
        alphas = [10**x for x in range(-16, -8)]
        fprimes = [grad]

        start_time = timeit.default_timer()

        for fprime in fprimes:
            for alpha in alphas:
                for i in range(maxiter):

                    """
                    w_true = np.zeros(w_dim)
                    w_true[:, :-1] = self.w
                    w_true[:, -1] = self.intercept
                    print loss(w_true, X, Y, alpha)
                    """

                    w0 = 1e-8 * np.random.randn(*w_dim)

                    """
                    print loss(w0, X, Y, alpha)
                    print logistic_grad(w0, X, Y, alpha)
                    print utils.approx_fprime_helper(w0.ravel(), loss, 1e-8,
                                                     args=(X, Y, alpha))
                    """

                    num_unknowns = len(w0.ravel())
                    method = "BFGS"
                    if num_unknowns >= 1000:
                        method = "L-BFGS-B"

                    print 'finding solution of system of {} equations with {}' \
                          ' unknowns with {}'.format(len(X), num_unknowns, method)

                    try:
                        optimLogitBFGS = minimize(loss, x0=w0,
                                                  method=method,
                                                  args=(X, Y, alpha),
                                                  jac=fprime,
                                                  options={'gtol': 1e-6,
                                                           'disp': True,
                                                           'maxiter': 100})
                        wopt = optimLogitBFGS.x
                    except ValueError:
                        wopt = np.zeros(w0.shape)

                    # reshape the coefficient vector
                    if k == 2:
                        int_opt = wopt[-1]
                        wopt = np.array([wopt[:-1]])
                    else:
                        wopt = wopt.reshape(k, -1)
                        int_opt = wopt[:, -1]
                        wopt = wopt[:, :-1]

                    # check the accuracy over the small set of training vectors
                    acc = self.evaluate(wopt, int_opt, X)
		    print 'obtained train accuracy of {}'.format(acc)
                    if acc > 0.99:
                        end_time = timeit.default_timer()
                        print >> sys.stderr, "opti ran for %.2f s" \
                                             % (end_time - start_time)
                        return wopt, int_opt
                    if acc >= best_acc:
                        best_acc = acc
                        best_w = wopt
                        best_int = int_opt

        end_time = timeit.default_timer()
        print >> sys.stderr, "opti ran for %.2f s" % (end_time - start_time)
        return best_w, best_int

    def find_coeffs(self, m, baseline=False, adapt=False):
        k = len(self.classes)       # number of classes
        n = self.num_features()     # vector dimension

        # generate random queries
        if not adapt:
            X = self.gen_query_set(n, test_size=m)
        else:
            X = utils.line_search_oracle(n, m, self.query, self.gen_query_set)

        self.X_train = X

        # get the probabilities for all queries
        if baseline:
            model = self.baseline_model(X)

            return model
        else:
            Y = self.query_probas(X)

        return self.select_and_run_opti(k, n, X, Y)

    def select_and_run_opti(self, k, n, X, Y):

        if self.multinomial:
            """
            Recover the full coefficient vector by minimizing the
            cross entropy loss.
            """
            wdim = (k, n + 1)
            wopt, int_opt = self.run_opti(multinomial_loss,
                                          multnomial_grad, X, Y, wdim)
        else:
            if k == 2:
                """
                Recover the single coefficient vector by minimizing the
                cross entropy loss.
                """
                wdim = (1, n + 1)
                wopt, int_opt = self.run_opti(logistic_loss,
                                              logistic_grad_bin, X, Y, wdim)
            else:
                """
                Recover the full coefficient vector by minimizing the
                cross entropy loss.
                """
                wdim = (k, n + 1)
                wopt, int_opt = self.run_opti(logistic_loss,
                                              logistic_grad, X, Y, wdim)

        return wopt, int_opt, len(X)
    
    def find_coeffs_bin(self, budget):
        k = len(self.classes)       # number of classes
        assert k == 2
        n = self.num_features()     # vector dimension

        X_train = self.gen_query_set(n, budget)
        y = logit(self.query_probas(X_train)[:, 1])

        X = np.hstack((X_train, np.ones((budget, 1))))

        if budget == n+1:
            try:
                w_opt = np.linalg.solve(X, y).T
            except np.linalg.linalg.LinAlgError:
                w_opt = np.linalg.lstsq(X, y)[0].T
        else:
            w_opt = np.linalg.lstsq(X, y)[0].T

        int_opt = w_opt[-1]
        w_opt = np.array([w_opt[:-1]])

        self.X_train = X_train

        return w_opt, int_opt

    def find_coeffs_adaptive(self, step, query_budget, baseline=False):
        assert query_budget > 0
        k = len(self.classes)       # number of classes
        n = self.num_features()     # vector dimension

        X = self.gen_query_set(n, test_size=step)

        while query_budget > 0:
            query_budget -= step
            # print 'training with {} queries'.format(len(X))
            if baseline:
                model = self.baseline_model(X)
            else:
                Y = self.query_probas(X)
                w_opt, int_opt, _ = self.select_and_run_opti(k, n, X, Y)

            if baseline:
                predict_func = lambda x: model.predict(x)
                predict_func_p = lambda x: model.predict_proba(x)
            else:
                predict_func = lambda x: predict_classes(x, w_opt, int_opt,
                                                         self.get_classes())
                predict_func_p = lambda x: predict_probas(x, w_opt, int_opt,
                                                          self.multinomial)

            if query_budget > 0:
                X_local = self.gen_query_set(n, test_size=query_budget)
                Y_local = predict_func(X_local)

                if len(pd.Series(Y_local[0:100]).unique()) == 1 \
                        or callable(getattr(self, 'encode', None)):
                    Y_local_p = predict_func_p(X_local)

                    if Y_local_p.ndim == 1 or Y_local_p.shape[1] == 1:
                        Y_local_p = np.hstack([1 - Y_local_p, Y_local_p])

                    Y_local_p.sort()
                    scores = Y_local_p[:, -1] - Y_local_p[:, -2]

                    adaptive_budget = (min(step, query_budget)*3)/4
                    random_budget = min(step, query_budget) - adaptive_budget

                    indices = scores.argsort()[0:adaptive_budget]
                    samples = X_local[indices, :]
                    X_random = self.gen_query_set(n, random_budget)
                    samples = np.vstack((samples, X_random))
                else:
                    # reserve some budget for random queries
                    adaptive_budget = (min(step, query_budget)*3)/4
                    adaptive_budget += adaptive_budget % 2
                    random_budget = min(step, query_budget) - adaptive_budget
                    samples = utils.line_search(X_local[0:100], Y_local[0:100],
                                                adaptive_budget/2,
                                                predict_func)
                    X_random = self.gen_query_set(n, random_budget)
                    samples = np.vstack((samples, X_random))

                assert len(samples) == min(step, query_budget)

                X = np.vstack((X, samples))

        if baseline:
            return model
        else:
            return w_opt, int_opt

    def evaluate(self, wopt, int_opt, X_test, base_model=None):
        # get the true class labels
        y_true = self.query(X_test)

        if X_test.shape[1] != self.num_features():
            X_test = self.encode(X_test)

        # predict classes using the optimized coefficients
        y_pred = predict_classes(X_test, wopt, int_opt, self.classes)

        """
        _, _, X, _, _ = utils.prepare_data(self.model_id, onehot=False)
        X = X.values
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                print y_true[i], y_pred[i], X[i]
        """

        if base_model is not None:

            y_pred_base = base_model.predict(X_test)

            return accuracy_score(y_true, y_pred), \
                   accuracy_score(y_true, y_pred_base)

        return accuracy_score(y_true, y_pred)

    def evaluate_probas(self, wopt, int_opt, X_test, base_model=None):
        # get the true class probas
        y_true = self.query_probas(X_test)

        if X_test.shape[1] != self.num_features():
            X_test = self.encode(X_test)

        # predict class probas using the optimized coefficients
        y_pred = predict_probas(X_test, wopt, int_opt, self.multinomial)

        if base_model is not None:
            y_pred_base = base_model.predict_proba(X_test)
            y_pred_base = pd.DataFrame(y_pred_base, columns=base_model.classes_)

            for col in self.classes:
                if col not in base_model.classes_:
                    y_pred_base[col] = 0

            y_pred_base.columns = self.classes

            return utils.stat_distance(y_pred, y_true), \
                   utils.stat_distance(y_pred_base.as_matrix(), y_true)

        return utils.stat_distance(y_pred, y_true)

    def evaluate_model(self, wopt, int_opt, base_model=None):
        try:
            w_true = self.w
            int_true = self.intercept

            if self.multinomial:
                w_true = w_true - w_true[0]
                wopt = wopt - wopt[0]
                int_true = int_true - int_true[0]
                int_opt = int_opt - int_opt[0]

            loss = np.sum(np.abs(w_true - wopt)) + \
                   np.sum(np.abs(int_true - int_opt))

            if base_model is not None:
                if isinstance(base_model, DummyClassifier) \
                        or len(self.get_classes()) != len(base_model.classes_):
                    loss_base = np.sum(np.abs(w_true)) + \
                                np.sum(np.abs(int_true))
                else:
                    w_base = base_model.coef_
                    int_base = base_model.intercept_

                    if self.multinomial:
                        w_base = w_base - w_true[0]
                        int_base = int_base - int_base[0]

                    loss_base = np.sum(np.abs(w_true - w_base)) + \
                                np.sum(np.abs(int_true - int_base))

                return loss, loss_base
            return loss

        except AttributeError:
            return np.nan

    def baseline_model(self, X):
        Y = pd.Series(self.query(X))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.multinomial:
                    model = LogisticRegression(solver='lbfgs',
                                               C=1e40,
                                               multi_class='multinomial',
                                               tol=1e-20, max_iter=10000)
                else:
                    model = LogisticRegression(solver='lbfgs',
                                               C=1e40,
                                               multi_class='ovr', tol=1e-20,
                                               max_iter=10000)

                model.fit(X, Y)

        except ValueError:
            model = DummyClassifier(strategy="stratified")
            model.fit(X, Y)

        return model

    def lowd_meek(self, budget, delta=1e-8):
        [c1, c0] = self.get_classes()

        n = self.num_features()
        m = 0
        x_pos = None
        x_neg = None

        #
        # Find a positive and a negative instance
        #
        while x_pos is None or x_neg is None:
            x = self.gen_query_set(n, 1)[0]
            c = self.query([x])[0]
            if c == c0:
                x_pos = x
            else:
                x_neg = x
            m += 1

        s_pos = x_pos.copy()
        s_neg = None
        f = None
        #
        # Find a sign witness
        #
        for i in range(n):
            s_pos_old = s_pos.copy()
            s_pos[i] = x_neg[i]
            m += 1

            if self.query([s_pos])[0] == c1:
                s_neg = s_pos.copy()
                s_pos = s_pos_old
                f = i
                break

        if m >= budget:
            return np.zeros((1,n)), 0, m

        curr_m = m

        for eps in [10.0**x for x in range(-6, 1)]:
            m = curr_m

            w = np.zeros(n)
            w[f] = (s_pos[f] - s_neg[f])/abs(s_pos[f] - s_neg[f])

            x = s_pos.copy()
            # binary search on feature f
            while abs(s_pos[f] - s_neg[f]) >= eps/4:
                x[f] = 0.5 * (s_pos[f] + s_neg[f])
                c = self.query([x])
                if c == c1:
                    s_neg[f] = x[f]
                else:
                    s_pos[f] = x[f]
                m += 1

            x = s_neg
            x[f] -= w[f]

            # search other features
            for i in range(n):
                if i == f:
                    continue

                u = np.zeros(n)
                u[i] = 1.0/delta
                test = self.query([x + u, x - u])
                m += 2
                if test[0] == test[1]:
                    w[i] = 0.0
                else:
                    step = 1
                    x_0 = x.copy()
                    x_1 = x.copy()

                    assert self.query([x_0])[0] == c1
                    #print 'x_0 = {}'.format(x_0)
                    # exponential search
                    while self.query([x_0]) != c0:
                        #print 'query of {} = {}'.format(x_0, c1)

                        if step > 0:
                            step *= -1
                        else:
                            step *= -2

                        if step != -1:
                            x_1[i] = step/2 * x[i]
                        x_0[i] = step * x[i]
                        m += 1

                    #print 'x_0 = {}'.format(x_0)
                    #print 'x_1 = {}'.format(x_1)
                    #print self.query([x_0, x_1])

                    assert list(self.query([x_0, x_1])) == [c0, c1]

                    mid = x_0.copy()

                    # binary search
                    while abs(x_0[i] - x_1[i]) >= eps/4:
                        mid[i] = 0.5 * (x_0[i] + x_1[i])
                        c = self.query([mid])
                        if c == c1:
                            x_1[i] = mid[i]
                        else:
                            x_0[i] = mid[i]
                        m += 1

                    w[i] = 1.0 / (x_1[i] - x[i])

            intercept = -np.dot(x_1, w)

            if m <= budget:
                return np.array([w]), intercept, m

        return np.zeros((1, n)), 0, m

    def run(self, data, X_test, test_size=100000, random_seed=0,
            alphas=[0.5, 1, 2, 5, 10, 20, 50, 100],
            methods=["passive", "adapt-local", "adapt-oracle"],
            baseline=True):

        np.random.seed(random_seed)

        print ','.join(['%s'] * 9) \
              % ('dataset', 'method', 'budget', 'mode', 'loss',
                 'loss_u', 'probas', 'probas_u', 'model l1')

        # number of unknown coefficients
        k = len(self.get_classes())
        n = self.num_features()
        num_unknowns = (k - int(k == 2)) * (n + 1)
        #num_unknowns = n+1

        X_test_u = self.gen_query_set(n, test_size, force_input_space=True)

        if k == 2:
            for alpha in alphas:
                m = int(alpha * num_unknowns)

                w_opt, int_opt = self.find_coeffs_bin(budget=m)

                # compute the accuracy of the predictions
                if X_test is not None:
                    acc = self.evaluate(w_opt, int_opt, X_test)
                    l1 = self.evaluate_probas(w_opt, int_opt, X_test)
                else:
                    acc, l1 = [np.nan] * 2

                acc_u = self.evaluate(w_opt, int_opt, X_test_u)
                l1_u = self.evaluate_probas(w_opt, int_opt, X_test_u)
                loss = self.evaluate_model(w_opt, int_opt)

                if X_test_u.shape[1] == 2 \
                        and callable(getattr(self, 'encode', None)):
                    print X_test_u
                    utils.plot_decision_boundary(
                        lambda x: predict_classes(self.encode(x), w_opt,
                                                  int_opt, self.get_classes()),
                        X_test_u,
                        self.query(X_test_u),
                        bounds=[-1, 1, -1, 1]
                    )

                print '%s,%s,%d,extr,%.2e,%.2e,%.2e,%.2e,%.2e' % \
                      (data, 'binary', m, 1-acc, 1-acc_u,
                       l1, l1_u, loss)

                if baseline and not callable(getattr(self, 'encode', None)):
                    w_base, int_base, m_base = self.lowd_meek(budget=m)

                    # compute the accuracy of the predictions
                    if X_test is not None:
                        acc_base = self.evaluate(w_base, int_base, X_test)
                        l1_base = self.evaluate_probas(w_base, int_base, X_test)
                    else:
                        acc_base, l1_base = [np.nan] * 2

                    acc_u_base = self.evaluate(w_base, int_base, X_test_u)
                    l1_u_base = self.evaluate_probas(w_base, int_base, X_test_u)
                    loss_base = self.evaluate_model(w_base, int_base)

                    print '%s,%s,%d,base,%.2e,%.2e,%.2e,%.2e,%.2e' % \
                          (data, 'lowd-meek', m,
                           1-acc_base, 1-acc_u_base, l1_base, l1_u_base,
                           loss_base)

        for alpha in alphas:
            m = int(alpha * num_unknowns)
            step = (m + 4)/5

            for method in methods:

                if m < 5:
                    print '%s,%s,%d,extr,%.2e,%.2e,%.2e,%.2e,%.2e' % \
                          (data, method, m, 1.0, 1.0, 1.0, 1.0, 1.0)

                    print '%s,%s,%d,base,%.2e,%.2e,%.2e,%.2e,%.2e' % \
                          (data, method, m, 1.0, 1.0, 1.0, 1.0, 1.0)
                    continue

                base_model = None
                if method == "passive":
                    w_opt, int_opt, m = self.find_coeffs(m)

                    if baseline:
                        base_model = self.find_coeffs(m, baseline=True)
                elif method == "adapt-local":
                    w_opt, int_opt = self.find_coeffs_adaptive(step, m)

                    if baseline:
                        base_model = self.find_coeffs_adaptive(step, m,
                                                               baseline=True)
                elif method == "adapt-oracle":
                    w_opt, int_opt, m = self.find_coeffs(m, adapt=True)

                    if baseline:
                        base_model = self.find_coeffs(m, baseline=True,
                                                      adapt=True)

                # compute the accuracy of the predictions
                if X_test is not None:
                    acc = self.evaluate(w_opt, int_opt, X_test,
                                        base_model=base_model)
                    l1 = self.evaluate_probas(w_opt, int_opt, X_test,
                                              base_model=base_model)
                else:
                    if baseline:
                        acc = [np.nan] * 2
                        l1 = [np.nan] * 2
                    else:
                        acc, l1 = np.nan, np.nan

                acc_u = self.evaluate(w_opt, int_opt, X_test_u,
                                      base_model=base_model)

                l1_u = self.evaluate_probas(w_opt, int_opt, X_test_u,
                                            base_model=base_model)

                loss = self.evaluate_model(w_opt, int_opt,
                                           base_model=base_model)

                if baseline:
                    print '%s,%s,%d,extr,%.2e,%.2e,%.2e,%.2e,%.2e' % \
                          (data, method, m, 1-acc[0], 1-acc_u[0], l1[0],
                           l1_u[0], loss[0])

                    print '%s,%s,%d,base,%.2e,%.2e,%.2e,%.2e,%.2e' % \
                          (data, method, m, 1-acc[1], 1-acc_u[1], l1[1],
                           l1_u[1], loss[1])
                else:
                    print '%s,%s,%d,extr,%.2e,%.2e,%.2e,%.2e,%.2e' % \
                          (data, method, m, 1-acc, 1-acc_u, l1, l1_u, loss)

                if X_test_u.shape[1] == 2 \
                        and callable(getattr(self, 'encode', None)):
                    utils.plot_decision_boundary(
                        lambda x: predict_classes(self.encode(x), w_opt,
                                                  int_opt, self.get_classes()),
                        X_test_u,
                        self.query(X_test_u),
                        bounds=[-1, 1, -1, 1]
                    )
