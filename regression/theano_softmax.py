import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
from collections import OrderedDict
import cPickle
import pandas as pd
from sklearn.metrics import accuracy_score
import utils
import abc
import glob
from pylearn2.utils import serial

try:
    import matplotlib.pyplot as plt
except:
    plt = None 
    pass
import math


class SoftmaxLayer(object):
    def __init__(self, input, n_in, n_out, W=None, b=None):

        if W is None:
            # initialize with 0 the weights W

            W = theano.shared(
                value=numpy.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )

        if b is None:
            # initialize the biases b as a vector of n_out 0s
            b = theano.shared(
                value=numpy.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        self.b = b

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.y_probas = self.p_y_given_x

        self.input = input
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x) * y)

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class Softmax(object):

    def __init__(self, input, n_in, n_out, W=None, b=None):

        self.softmaxLayer = SoftmaxLayer(
            input=input,
            n_in=n_in,
            n_out=n_out,
            W=W,
            b=b
        )

        self.L2_sqr = (
            (self.softmaxLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.softmaxLayer.negative_log_likelihood
        )

        self.errors = self.softmaxLayer.errors
        self.y_pred = self.softmaxLayer.y_pred
        self.y_probas = self.softmaxLayer.y_probas
        self.params = self.softmaxLayer.params
        self.input = input


def apply_nesterov_momentum(updates, params, g):
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow=True)

        y = theano.shared(numpy.zeros(value.shape, dtype=value.dtype),
                          broadcastable=param.broadcastable)

        updates[y] = updates[param]
        updates[param] = (1 - g) * updates[param] + g * y

    return updates


def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy

    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)

    return shared_x, shared_y


def predict(model, test_set_x):
    # compile a predictor function
    predict_model = theano.function(
        inputs=[model.input],
        outputs=model.y_pred)

    predicted_values = predict_model(test_set_x)
    return predicted_values


def predict_probas(model, test_set_x):
    # compile a predictor function
    predict_model = theano.function(
        inputs=[model.input],
        outputs=model.y_probas)

    predicted_values = predict_model(test_set_x)
    return predicted_values


def calculate_loss(model, X, y, reg):
    y_pred = predict_probas(model, X)

    return - numpy.mean(y * numpy.log(y_pred)) + reg * model.L2_sqr.eval()


def save(model, filename):
    with open(filename, 'wb') as f:
        cPickle.dump([model.params], f, cPickle.HIGHEST_PROTOCOL)


def load(filename):
    x = T.matrix('x')

    with open(filename, 'rb') as f:
        [params] = cPickle.load(f)

    W, b = params

    n_in, n_out = W.get_value().shape

    # construct the MLP class
    classifier = Softmax(
        input=x,
        n_in=n_in,
        n_out=n_out,
        W=W,
        b=b
    )

    classifier.softmaxLayer.W = W
    classifier.softmaxLayer.b = b

    return classifier


i = 0


def build_model(X, y, epsilon=1e-5,
                reg_lambda=0.0001, num_passes=10000, eps_factor=0.99,
                epoch=1000, print_loss=False, print_epoch=1000,
                batch_size=None, warm_start=None):

    n_in = X.shape[1]
    n_out = y.shape[1]

    data_x = theano.shared(numpy.asarray(X, dtype=theano.config.floatX),
                           borrow=True)
    data_y = theano.shared(numpy.asarray(y, dtype=theano.config.floatX),
                           borrow=True)

    if batch_size is None:
        batch_size = max(1, len(X)/1000)

    # compute number of minibatches
    n_batches = data_x.get_value(borrow=True).shape[0] / batch_size

    ###############
    # BUILD MODEL #
    ###############
    print '... building model ' \
          '(len(X)={}, batch_size={}, eps={}, eps_factor={})'.\
        format(len(X), batch_size, epsilon, eps_factor)

    # allocate symbolic variables for the data
    minibatch_offset = T.lscalar()  # index to a minibatch
    index = T.lscalar()  # index to a minibatch
    x_var = T.matrix('x_var')
    y_var = T.matrix('y_var')

    if warm_start is None:
        # construct the MLP class
        classifier = Softmax(
            input=x_var,
            n_in=n_in,
            n_out=n_out,
        )
    else:
        classifier = Softmax(
            input=x_var,
            n_in=n_in,
            n_out=n_out,
            W=warm_start.softmaxLayer.W,
            b=warm_start.softmaxLayer.b
        )

    # the cost we minimize during training
    cost = (
        classifier.negative_log_likelihood(y_var) +
        reg_lambda * classifier.L2_sqr
    )

    # compute the gradient of cost
    gparams = [T.grad(cost, param) for param in classifier.params]

    l_r = T.scalar('l_r', dtype=theano.config.floatX)
    updates = [
        (param, param - l_r * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    mu = T.scalar('mu', dtype=theano.config.floatX)
    updates_momentum = apply_nesterov_momentum(updates, classifier.params, mu)

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index, l_r, mu],
        outputs=cost,
        updates=updates_momentum,
        givens={
            x_var: data_x[index * batch_size: (index + 1) * batch_size],
            y_var: data_y[index * batch_size: (index + 1) * batch_size]
        },
    )

    grad_norm = theano.function(
        inputs=[],
        outputs=T.sqrt(T.sum([T.sum(T.sqr(g)) for g in gparams])),
        givens={
            x_var: data_x,
            y_var: data_y
        })

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    last_cost = 10*len(X)
    start_time = timeit.default_timer()
    i = 0

    l = 1
    g = 0

    while i < num_passes:
        i += 1
        for minibatch_index in xrange(n_batches):
            train_model(minibatch_index, epsilon, g)

        l_prev = l
        l = 0.5*(1 + numpy.sqrt(1 + 4 * l**2))
        g = (1 - l_prev)/l

        curr_cost = calculate_loss(classifier, X, y, reg_lambda)

        if curr_cost > last_cost:
            l = 1
            g = 0
            epsilon *= 0.5

        last_cost = curr_cost

        gnorm = grad_norm()

        if gnorm < 1e-8:
            break

        if i % print_epoch == 0:
            print 'Epoch %i: Cost: %f' % (i, curr_cost)
            print >> sys.stderr, 'Epoch %i: Cost: %f' % (i, curr_cost)

        if i % epoch == 0:
            epsilon *= eps_factor

    end_time = timeit.default_timer()
    print 'Optimization complete.'
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    return classifier


class SoftmaxExtractor(object):
    def __init__(self, dataset):
        self.dataset = dataset

    @abc.abstractmethod
    def num_features(self):
        return

    @abc.abstractmethod
    def get_classes(self):
        return

    @abc.abstractmethod
    def query_probas(self, X):
        return

    @abc.abstractmethod
    def query(self, X):
        return

    def extract(self, budget, steps=[],
                adaptive_oracle=False, baseline=False,
                epsilon=1e-2, reg_lambda=1e-16,  eps_factor=0.99,
                epoch=100, print_epoch=10, batch_size=1, num_passes=1000,
                random_seed=0):

        numpy.random.seed(random_seed)

        assert not (adaptive_oracle and steps)

        if steps:
            step = steps[0]
        else:
            step = budget

        if not adaptive_oracle:
            X_ext = utils.gen_query_set(self.num_features(), step)
        else:
            X_ext = utils.line_search_oracle(self.num_features(), step,
                                             self.query)

        model = None
        idx = 0
        while budget > 0:
            idx += 1
            budget -= step

            y_ext = self.query(X_ext)

            if not baseline:
                y_ext_p = self.query_probas(X_ext)
            else:
                num_classes = len(self.get_classes())
                y_ext_p = numpy.zeros((len(y_ext), num_classes))
                y_ext_p[numpy.arange(len(y_ext)), y_ext] = 1

            print y_ext_p

            print '{}'.format(- numpy.mean(y_ext_p * numpy.log(y_ext_p)))

            model = build_model(X_ext, y_ext_p,
                                epsilon=epsilon, reg_lambda=reg_lambda,
                                num_passes=num_passes, eps_factor=eps_factor,
                                epoch=epoch, print_epoch=print_epoch,
                                batch_size=batch_size, warm_start=model)

            mtype = "base" if baseline else "extr"
            mode = "adapt-local" if steps \
                else "adapt-oracle" if adaptive_oracle \
                else "passive"
            save(model, 'experiments/inversion/{}/models/{}_{}_{}.pkl'.
                 format(self.dataset, mode, mtype, len(X_ext)))

            if budget > 0:
                step = steps[idx] - steps[idx-1]
                X_local = utils.gen_query_set(n=X_repr.shape[1], test_size=1000)
                Y_local = predict(model, X_local)

                assert len(pd.Series(Y_local).unique()) != 1

                adaptive_budget = (min(step, budget)*3)/4
                adaptive_budget += adaptive_budget % 2
                random_budget = min(step, budget) - adaptive_budget

                predict_func = lambda x: predict(model, x)
                samples = utils.line_search(X_local, Y_local, adaptive_budget/2,
                                            predict_func)
                X_random = utils.gen_query_set(X_ext.shape[1], random_budget)
                X_ext = numpy.vstack((samples, X_random, X_ext))


    def compare(self, X_test, X_unif, scaler=None):
        if X_test is not None:
            y = self.query(X_test)
            p = self.query_probas(X_test)

        y_u = self.query(X_unif)
        p_u = self.query_probas(X_unif)

        for mtype in ['extr', 'base']:
            for mode in ['passive', 'adapt-local', 'adapt-oracle']:
                files = glob.glob('experiments/inversion/{}/models/{}_{}_[0-9]*.pkl'.
                                  format(self.dataset, mode, mtype))

                sorted_f = sorted(files, key=lambda f: int(f.split('_')[-1].
                                                           split('.')[0]))

                for filename in sorted_f:
                    samples = filename.split('_')[-1].split('.')[0]
                    model_ext = load(filename)

                    if X_test is not None:
                        y_pred = predict(model_ext, X_test)
                        acc = accuracy_score(y, y_pred)

                        p_pred = predict_probas(model_ext, X_test)
                        l1 = utils.stat_distance(p, p_pred)
                    else:
                        acc, l1 = [numpy.nan] * 2

                    # compute the accuracy of the predictions on a uniform
                    # test set

                    y_pred_u = predict(model_ext, X_unif)
                    acc_u = accuracy_score(y_u, y_pred_u)

                    p_pred_u = predict_probas(model_ext, X_unif)
                    l1_u = utils.stat_distance(p_u, p_pred_u)

                    l1_model = numpy.nan

                    print '%s,%s,%d,%s,%.2e,%.2e,%.2e,%.2e,%.2e' % \
                          (self.dataset, mode, int(samples), mtype, 1-acc,
                           1-acc_u, l1, l1_u, l1_model)

if __name__ == '__main__':
    build_model()
