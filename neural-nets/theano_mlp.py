import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
from collections import OrderedDict
import cPickle
import abc
import utils
from sklearn.metrics import accuracy_score
import pandas as pd
import glob


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=T.tanh,
                 W=None, b=None):

        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class LogisticRegression(object):

    def __init__(self, input, n_in, n_out, W=None, b=None):

        if W is None:
            # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
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

        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x) * y)

    def errors(self, y):

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class MLP(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out,
                 W1=None, b1=None, W2=None, b2=None, force_reg=False):

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh,
            W=W1,
            b=b1
        )

        if force_reg:
            # The logistic regression layer gets as input the hidden units
            # of the hidden layer
            self.logRegressionLayer = LogisticRegression(
                input=input,
                n_in=n_in,
                n_out=n_out,
                W=W2,
                b=b2
            )

            # regularizer
            self.L2_sqr = (
                (self.logRegressionLayer.W ** 2).sum()
            )

            # the parameters of the model are the parameters of the two layer
            self.params = self.logRegressionLayer.params

        else:
            # The logistic regression layer gets as input the hidden units
            # of the hidden layer
            self.logRegressionLayer = LogisticRegression(
                input=self.hiddenLayer.output,
                n_in=n_hidden,
                n_out=n_out,
                W=W2,
                b=b2
            )

            # regularizer
            self.L2_sqr = (
                (self.hiddenLayer.W ** 2).sum() +
                (self.logRegressionLayer.W ** 2).sum()
            )

            # the parameters of the model are the parameters of the two layer
            self.params = self.hiddenLayer.params + \
                          self.logRegressionLayer.params

        # negative log likelihood of the MLP
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        self.errors = self.logRegressionLayer.errors
        self.y_pred = self.logRegressionLayer.y_pred
        self.y_probas = self.logRegressionLayer.y_probas
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
        outputs=model.logRegressionLayer.y_pred)

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


def load(filename, force_reg=False):
    x = T.matrix('x')  # the data is presented as rasterized images
    rng = numpy.random.RandomState(1234)

    with open(filename, 'rb') as f:
        [params] = cPickle.load(f)

    if force_reg:
        W2, b2 = params
        n_in, n_out = W2.get_value().shape
        W1 = None
        b1 = None
        n_hidden = 1
    else:
        W1, b1, W2, b2 = params
        n_hidden, n_out = W2.get_value().shape
        n_in = W1.get_value().shape[0]


    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=n_out,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        force_reg=force_reg
    )

    #classifier.hiddenLayer.W = W1
    #classifier.hiddenLayer.b = b1
    #classifier.logRegressionLayer.W = W2
    #classifier.logRegressionLayer.b = b2

    return classifier


def build_model(nn_hdim, X, y, epsilon=1e-5, reg_lambda=0.0001, num_passes=1000,
                eps_factor=0.99, epoch=1000, print_loss=False, print_epoch=1000,
                batch_size=None, force_reg=False):

    n_in = X.shape[1]
    n_out = y.shape[1]

    data_x = theano.shared(numpy.asarray(X, dtype=theano.config.floatX),
                           borrow=True)
    data_y = theano.shared(numpy.asarray(y, dtype=theano.config.floatX),
                           borrow=True)

    if batch_size is None or batch_size > len(X):
        batch_size = len(X)

    batch_size = max(batch_size, len(X)/1000)

    # compute number of minibatches
    n_batches = data_x.get_value(borrow=True).shape[0] / batch_size

    ###############
    # BUILD MODEL #
    ###############
    print '... building model ' \
          '(hdim={}, len(X)={}, batch_size={}, eps={}, eps_factor={})'.\
        format(nn_hdim, len(X), batch_size, epsilon, eps_factor)
    print >> sys.stderr, '... building model (hdim={}, len(X)={}, ' \
                         'batch_size={}, eps={}, eps_factor={})'.\
        format(nn_hdim, len(X), batch_size, epsilon, eps_factor)

    # allocate symbolic variables for the data
    index = T.lscalar()
    x_var = T.matrix('x_var')
    y_var = T.matrix('y_var')

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x_var,
        n_in=n_in,
        n_hidden=nn_hdim,
        n_out=n_out,
        force_reg=force_reg
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

        if gnorm < 1e-5:
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


class PerceptronExtractor(object):

    def __init__(self, dataset, hidden_nodes):
        self.dataset = dataset
        self.hidden_nodes = hidden_nodes

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

    @abc.abstractmethod
    def calculate_loss(self, X, y, reg):
        return

    def extract(self, X_train, y_train, budget, steps=[],
                adaptive_oracle=False, baseline=False, epsilon=1e-1,
                num_passes=1000, reg_lambda=1e-8, eps_factor=0.99, epoch=100,
                print_loss=True, print_epoch=10, batch_size=20, random_seed=0):

        numpy.random.seed(random_seed)

        assert not (adaptive_oracle and steps)

        if steps:
            step = steps[0]
        else:
            step = budget

        if not adaptive_oracle:
            X_ext = utils.gen_query_set(X_train.shape[1], step)
        else:
            X_ext = utils.line_search_oracle(X_train.shape[1], step, self.query)

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

            # Loss with correct parameters:
            print self.calculate_loss(X_ext, y_ext_p, reg_lambda)
            print >> sys.stderr, self.calculate_loss(X_ext, y_ext_p, reg_lambda)

            model = build_model(self.hidden_nodes, X_ext, y_ext_p,
                                epsilon=epsilon, num_passes=num_passes,
                                reg_lambda=reg_lambda, epoch=epoch,
                                eps_factor=eps_factor, print_loss=print_loss,
                                print_epoch=print_epoch, batch_size=batch_size)

            m_type = "base" if baseline else "extr"
            mode = "adapt-local" if steps \
                else "adapt-oracle" if adaptive_oracle \
                else "passive"

            save(model, 'experiments/{}/models/{}_{}_{}_{}_{}.pkl'.
                 format(self.dataset, mode, m_type,
                        self.hidden_nodes, len(X_ext), random_seed))

            if X_train is not None and X_train.shape[1] == 2:
                bounds = [-1.1, 1.1, -1.1, 1.1]
                filename = 'experiments/{}/plots/{}_{}_{}_{}_{}_boundary'.\
                    format(self.dataset, mode, m_type,
                           self.hidden_nodes, len(X_ext), random_seed)
                utils.plot_decision_boundary(lambda x: predict(model, x),
                                             X_train.values, y_train,
                                             bounds, filename)

                filename = 'experiments/{}/plots/{}_{}_{}_{}_{}_boundary_ext'.\
                    format(self.dataset, mode, m_type,
                           self.hidden_nodes, len(X_ext), random_seed)
                utils.plot_decision_boundary(lambda x: predict(model, x),
                                             X_ext, y_ext, bounds, filename)

            if budget > 0:
                step = steps[idx] - steps[idx-1]
                X_local = utils.gen_query_set(n=X_ext.shape[1], test_size=1000)
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

    def compare(self, X_test, X_unif, force_reg=False):

        y = self.query(X_test)
        y_u = self.query(X_unif)
        p = self.query_probas(X_test)
        p_u = self.query_probas(X_unif)

        for mtype in ['extr', 'base']:
            for mode in ['passive', 'adapt-local', 'adapt-oracle']:
                files = glob.glob(
                    'experiments/{}/models/{}_{}_{}_[0-9]*_[0-9]*.pkl'.
                        format(self.dataset, mode, mtype, self.hidden_nodes))

                sorted_f = sorted(files, key=lambda f: int(f.split('_')[-2]))

                for filename in sorted_f:
                    seed = filename.split('_')[-1].split('.')[0]
                    samples = filename.split('_')[-2]
                    model_ext = load(filename)

                    if X_test.shape[1] == 2:
                        filename = 'experiments/{}/plots/{}_{}_{}_{}_{}_comp'.\
                            format(self.dataset, mode, mtype,
                                   self.hidden_nodes, samples, seed)
                        utils.compare_decision_boundary(
                            lambda x: self.query(x),
                            lambda x: predict(model_ext, x),
                            X_test.values, filename)

                    # test set accuracy
                    y_pred = predict(model_ext, X_test)
                    acc = accuracy_score(y, y_pred)

                    y_pred_u = predict(model_ext, X_unif)
                    acc_u = accuracy_score(y_u, y_pred_u)

                    p_pred = predict_probas(model_ext, X_test)
                    l1 = utils.stat_distance(p, p_pred)

                    p_pred_u = predict_probas(model_ext, X_unif)
                    l1_u = utils.stat_distance(p_u, p_pred_u)

                    try:
                        l1_model = self.compare_models(model_ext, force_reg)
                    except IOError:
                        l1_model = numpy.nan

                    print '%s,%s,%d,%s,%.2e,%.2e,%.2e,%.2e,%.2e' % \
                          (self.dataset, mode, int(samples), mtype, 1-acc,
                           1-acc_u, l1, l1_u, l1_model)

    def compare_models(self, model_ext, force_reg):
        if force_reg:
            return 1

        model_bb = load('experiments/{}/models/oracle_{}.pkl'.
                        format(self.dataset, self.hidden_nodes))
        W1_true = model_bb.hiddenLayer.W.eval()
        b1_true = model_bb.hiddenLayer.b.eval()
        W2_true = model_bb.logRegressionLayer.W.eval()
        b2_true = model_bb.logRegressionLayer.b.eval()

        # normalize softmax
        W2_true -= W2_true[0, :]
        b2_true -= b2_true[0]

        W1_ext = model_ext.hiddenLayer.W.eval()
        b1_ext = model_ext.hiddenLayer.b.eval()
        W2_ext = model_ext.logRegressionLayer.W.eval()
        b2_ext = model_ext.logRegressionLayer.b.eval()

        m1 = numpy.vstack((W1_true, b1_true)).T
        m2 = numpy.vstack((W1_ext, b1_ext)).T

        hidden_loss, matching, neg = utils.min_l1_dist(m1, m2)

        W2_ext_old = W2_ext.copy()

        for (i, j) in matching:
            mul = -1 if (i, j) in neg else 1
            W2_ext[i] = mul * W2_ext_old[j]

        W2_ext -= W2_ext[0, :]
        b2_ext -= b2_ext[0]

        loss = hidden_loss
        loss += numpy.sum(numpy.abs(W2_true - W2_ext))
        loss += numpy.sum(numpy.abs(b2_true - b2_ext))

        return loss


if __name__ == '__main__':
    build_model()
