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
try:
    import matplotlib.pyplot as plt
except:
    plt = None 
    pass
import math


class KernelLayer(object):
    def __init__(self, input, gamma, n_out, Y, learn_Y=True, W=None, b=None):

        self.gamma = gamma

        if not isinstance(Y, T.sharedvar.SharedVariable):
            Y = theano.shared(
                value=numpy.array(
                    Y,
                    dtype=theano.config.floatX
                ),
                name='Y',
                borrow=True
            )

        self.Y = Y

        if W is None:
            # initialize with 0 the weights W
            W = theano.shared(
                value=numpy.zeros(
                    (Y.get_value().shape[0], n_out),
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

        Y1 = Y[numpy.newaxis, :, :]
        X1 = input[:, numpy.newaxis, :]
        rbf = T.exp(-gamma * T.sum((Y1-X1)**2, axis=-1))

        self.p_y_given_x = T.nnet.softmax(T.dot(rbf, self.W) + self.b)

        """
        # One-vs-Rest
        self.p_y_given_x = T.nnet.sigmoid(T.dot(rbf, self.W) + self.b)
        self.p_y_given_x /= \
            self.p_y_given_x.sum(axis=1).\
                reshape((self.p_y_given_x.shape[0], 1))
        """

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.y_probas = self.p_y_given_x

        self.input = input

        if learn_Y:
            self.params = [self.Y, self.W, self.b]
        else:
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


class KernelLog(object):

    def __init__(self, input, gamma, n_out, Y, learn_Y=True, W=None, b=None):

        self.kernelLayer = KernelLayer(
            input=input,
            gamma=gamma,
            n_out=n_out,
            Y=Y,
            learn_Y=learn_Y,
            W=W,
            b=b
        )

        self.L2_sqr = (
            (self.kernelLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.kernelLayer.negative_log_likelihood
        )

        self.errors = self.kernelLayer.errors
        self.y_pred = self.kernelLayer.y_pred
        self.y_probas = self.kernelLayer.y_probas

        self.params = self.kernelLayer.params
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
        cPickle.dump([model.params, model.kernelLayer.Y,
                      model.kernelLayer.gamma], f, cPickle.HIGHEST_PROTOCOL)


def load(filename):
    x = T.matrix('x')

    with open(filename, 'rb') as f:
        [params, Y, gamma] = cPickle.load(f)

    W, b = params[-2:]

    _, n_out = W.get_value().shape

    # construct the MLP class
    classifier = KernelLog(
        input=x,
        gamma=gamma,
        n_out=n_out,
        Y=Y,
        W=W,
        b=b
    )

    classifier.kernelLayer.Y = Y
    classifier.kernelLayer.W = W
    classifier.kernelLayer.b = b

    return classifier


i = 0


def build_model(X_repr, learn_kernel, X, y, gamma=1, epsilon=1e-5,
                reg_lambda=0.0001, num_passes=10000, eps_factor=0.99,
                epoch=1000, print_loss=False, print_epoch=1000,
                batch_size=None, warm_start=None):

    nn_hdim = X_repr.shape[0]
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
          '(num_repr={}, len(X)={}, batch_size={}, eps={}, eps_factor={})'.\
        format(nn_hdim, len(X), batch_size, epsilon, eps_factor)

    # allocate symbolic variables for the data
    minibatch_offset = T.lscalar()  # index to a minibatch
    index = T.lscalar()  # index to a minibatch
    x_var = T.matrix('x_var')
    y_var = T.matrix('y_var')

    if warm_start is None:
        # construct the MLP class
        classifier = KernelLog(
            input=x_var,
            gamma=gamma,
            n_out=n_out,
            Y=X_repr,
            learn_Y=learn_kernel
        )
    else:
        classifier = KernelLog(
            input=x_var,
            gamma=gamma,
            n_out=n_out,
            Y=warm_start.kernelLayer.Y,
            learn_Y=learn_kernel,
            W=warm_start.kernelLayer.W,
            b=warm_start.kernelLayer.b
        )

    # the cost we minimize during training
    cost = (
        classifier.negative_log_likelihood(y_var) +
        reg_lambda * classifier.L2_sqr
    )

    """
    #  compile a theano function that returns the cost of a minibatch
    batch_cost = theano.function(
        [minibatch_offset],
        cost,
        givens={
            x_var: data_x[minibatch_offset: minibatch_offset + batch_size],
            y_var: data_y[minibatch_offset: minibatch_offset + batch_size]
        },
        name="batch_cost"
    )

    # compile a theano function that returns the gradient of the minibatch
    # with respect to theta
    batch_grad = theano.function(
        [minibatch_offset],
        [T.grad(cost, param).ravel() for param in classifier.params],
        givens={
            x_var: data_x[minibatch_offset: minibatch_offset + batch_size],
            y_var: data_y[minibatch_offset: minibatch_offset + batch_size]
        },
        name="batch_grad"
    )

    # creates a function that computes the average cost on the training set
    def train_fn(coeffs):
        offset = 0
        if learn_kernel:
            Y = coeffs[offset:offset+X_repr.shape[0]*X_repr.shape[1]].reshape((X_repr.shape[0], X_repr.shape[1]))
            offset += X_repr.shape[0] * X_repr.shape[1]
            classifier.kernelLayer.Y.set_value(Y, borrow=True)

        W = coeffs[offset:offset+nn_hdim*n_out].reshape((nn_hdim, n_out))
        offset += nn_hdim*n_out
        b = coeffs[offset:offset+n_out]
        classifier.kernelLayer.W.set_value(W, borrow=True)
        classifier.kernelLayer.b.set_value(b, borrow=True)

        train_losses = [batch_cost(i * batch_size)
                        for i in xrange(n_batches)]
        return numpy.mean(train_losses)

    # creates a function that computes the average gradient of cost with
    # respect to theta
    def train_fn_grad(coeffs):
        offset = 0
        if learn_kernel:
            Y = coeffs[offset:offset+X_repr.shape[0]*X_repr.shape[1]].reshape((X_repr.shape[0], X_repr.shape[1]))
            offset += X_repr.shape[0] * X_repr.shape[1]
            classifier.kernelLayer.Y.set_value(Y, borrow=True)

        W = coeffs[offset:offset+nn_hdim*n_out].reshape((nn_hdim, n_out))
        offset += nn_hdim*n_out
        b = coeffs[offset:offset+n_out]
        classifier.kernelLayer.W.set_value(W, borrow=True)
        classifier.kernelLayer.b.set_value(b, borrow=True)

        grad = numpy.hstack(batch_grad(0))
        for i in xrange(1, n_batches):
            grad += numpy.hstack(batch_grad(i * batch_size))
        return grad / n_batches

    # creates the validation function
    def callback(coeffs):
        global i
        offset = 0
        if learn_kernel:
            Y = coeffs[offset:offset+X_repr.shape[0]*X_repr.shape[1]].reshape((X_repr.shape[0], X_repr.shape[1]))
            offset += X_repr.shape[0] * X_repr.shape[1]
            classifier.kernelLayer.Y.set_value(Y, borrow=True)

        W = coeffs[offset:offset+nn_hdim*n_out].reshape((nn_hdim, n_out))
        offset += nn_hdim*n_out
        b = coeffs[offset:offset+n_out]
        classifier.kernelLayer.W.set_value(W, borrow=True)
        classifier.kernelLayer.b.set_value(b, borrow=True)
        curr_cost = calculate_loss(classifier, X, y, reg_lambda)

        if i % print_epoch == 0:
            print 'Epoch %i: Cost: %f' % (i, curr_cost)
        i += 1


    ###############
    # TRAIN MODEL #
    ###############

    # using scipy conjugate gradient optimizer
    import scipy.optimize
    start_time = timeit.default_timer()
    res = scipy.optimize.minimize(
        fun=train_fn,
        x0=numpy.hstack([param.eval().ravel() for param in classifier.params]),
        jac=train_fn_grad,
        callback=callback,
        tol=1e-20,
        options={
            'disp': 0,
            'maxiter': num_passes
        }
    )
    end_time = timeit.default_timer()
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % (end_time - start_time))

    offset = 0
    coeffs = res.x
    if learn_kernel:
            Y = coeffs[offset:offset+X_repr.shape[0]*X_repr.shape[1]].reshape((X_repr.shape[0], X_repr.shape[1]))
            offset += X_repr.shape[0] * X_repr.shape[1]
            classifier.kernelLayer.Y.set_value(Y, borrow=True)

    W = coeffs[offset:offset+nn_hdim*n_out].reshape((nn_hdim, n_out))
    offset += nn_hdim*n_out
    b = coeffs[offset:offset+n_out]
    classifier.kernelLayer.W.set_value(W, borrow=True)
    classifier.kernelLayer.b.set_value(b, borrow=True)

    return classifier
    """

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


class KernelRegressionExtractor(object):
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

    @abc.abstractmethod
    def calculate_loss(self, X, y, reg):
        return

    def extract(self, X_train, y_train, num_repr, budget, steps=[],
                adaptive_oracle=False, baseline=False,
                gamma=1, epsilon=1e-2, reg_lambda=1e-16,  eps_factor=0.99,
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

        X_repr = utils.gen_query_set(self.num_features(), num_repr)

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

            print '{} ({})'.format(
                self.calculate_loss(X_ext, y_ext_p, reg_lambda),
                self.calculate_loss(X_ext, y_ext_p, 0))
            print >> sys.stderr, self.calculate_loss(X_ext, y_ext_p, reg_lambda)

            model = build_model(X_repr, False, X_ext, y_ext_p, gamma=gamma,
                                epsilon=epsilon, reg_lambda=reg_lambda,
                                num_passes=num_passes, eps_factor=eps_factor,
                                epoch=epoch, print_epoch=print_epoch,
                                batch_size=batch_size, warm_start=model)

            mtype = "base" if baseline else "extr"
            mode = "adapt-local" if steps \
                else "adapt-oracle" if adaptive_oracle \
                else "passive"
            save(model, 'experiments/KLR/{}/models/{}_{}_{}.pkl'.
                 format(self.dataset, mode, mtype, len(X_ext)))

            if X_train is not None and X_train.shape[1] == 2:
                bounds = [-1.1, 1.1, -1.1, 1.1]
                filename = 'experiments/KLR/{}/plots/{}_{}_{}_{}_boundary'.\
                    format(self.dataset, mode, mtype, len(X_ext), random_seed)
                utils.plot_decision_boundary(lambda x: predict(model, x),
                                             X_train.values, y_train,
                                             bounds, filename)

                filename = 'experiments/KLR/{}/plots/{}_{}_{}_{}_boundary_ext'.\
                    format(self.dataset, mode, mtype, len(X_ext), random_seed)
                utils.plot_decision_boundary(lambda x: predict(model, x),
                                             X_ext, y_ext, bounds, filename)

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

    def compare_models(self, model_ext, samples, verbose=True, scaler=None):
        model_bb = \
            load('experiments/KLR/{}/models/oracle.pkl'.format(self.dataset))

        numpy.set_printoptions(threshold='nan')

        oracle_repr = numpy.array(model_bb.kernelLayer.Y.eval())

        if verbose:
            print 'Oracle representers:'
            if scaler is not None:
                oracle_repr = numpy.round(
                    numpy.array(
                        sorted(
                            scaler.inverse_transform(oracle_repr)
                            , key=lambda d: tuple(d)
                        )
                    ), 1)
            print numpy.round(oracle_repr[0:20], 2)

        ext_repr = numpy.array(model_ext.kernelLayer.Y.eval())

        if verbose:
            print 'representers:'
            if scaler is not None:
                ext_repr = numpy.round(
                    numpy.array(
                        sorted(
                            scaler.inverse_transform(
                                model_ext.kernelLayer.Y.eval()),
                            key=lambda d: tuple(d)
                        )
                    ), 1)
            print numpy.round(ext_repr, 2)

            found = 0
            for x in ext_repr:
                if x.tolist() in oracle_repr.tolist():
                    print 'FOUND: {}'.format(x)
                    found += 1

            print 'found {}/{} representers'.format(found, len(oracle_repr))

        print self.query(ext_repr)
        print predict(model_ext, ext_repr)

        repr_loss = 1

        if len(oracle_repr) == len(ext_repr):
            repr_loss, matching = utils.min_l1_dist(oracle_repr, ext_repr)

            ext_repr_old = ext_repr.copy()
            for (i, j) in matching:
                ext_repr[i] = ext_repr_old[j]

        if verbose and 'digits' in self.dataset:
            side = math.sqrt(ext_repr.shape[1])

            fig_tot = plt.figure()
            for i in range(len(oracle_repr))[0:20]:
                ax_tot = plt.subplot(5, len(oracle_repr)/5, i + 1)
                ax_tot.axis('off')
                image = oracle_repr[i, :].reshape((side, side))
                ax_tot.imshow(image, cmap=plt.cm.gray_r,
                              interpolation='nearest')

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.axis('off')
                ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
                fig.savefig('experiments/KLR/{}/plots/oracle_repr_{}.png'.
                            format(self.dataset, i))
                plt.close(fig)

            fig_tot.savefig('experiments/KLR/{}/plots/oracle_repr.png'.
                            format(self.dataset))
            plt.close(fig_tot)

            oracle_repr_classes = self.query(oracle_repr)

            for c in self.get_classes():
                reprs = [repr for (repr, c2)
                         in zip(oracle_repr, oracle_repr_classes) if c2 == c]

                avg_repr = numpy.average(reprs, axis=0)
                image = avg_repr.reshape((side, side))
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.axis('off')
                ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
                fig.savefig('experiments/KLR/{}/plots/oracle_avg_{}.png'.
                            format(self.dataset, c))
                plt.close(fig)

            fig_tot = plt.figure()
            for i in range(len(ext_repr)):
                ax_tot = plt.subplot(5, len(ext_repr)/5, i + 1)
                ax_tot.axis('off')
                image = ext_repr[i, :].reshape((side, side))
                ax_tot.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.axis('off')
                ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
                fig.savefig('experiments/KLR/{}/plots/extr_{}_repr_{}.png'.
                            format(self.dataset, samples, i))
                plt.close(fig)

            fig_tot.savefig('experiments/KLR/{}/plots/extr_{}_repr.png'.
                            format(self.dataset, samples, i))
            plt.close(fig_tot)

        if verbose and 'faces' in self.dataset:
            side = math.sqrt(ext_repr.shape[1])

            plt.figure()
            for i in range(len(oracle_repr)):
                plt.subplot(5, len(oracle_repr)/5, i + 1)
                plt.axis('off')
                image = oracle_repr[i, :].reshape((side, side))
                plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.show()

            plt.figure()
            for i in range(len(ext_repr)):
                plt.subplot(5, len(ext_repr)/5, i + 1)
                plt.axis('off')
                image = ext_repr[i, :].reshape((side, side))
                plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.show()
        """
        W_true = model_bb.kernelLayer.W.eval()
        b_true = model_bb.kernelLayer.b.eval()

        # normalize softmax
        W_true -= W_true[0, :]
        b_true -= b_true[0]

        W_ext = model_ext.kernelLayer.W.eval()
        b_ext = model_ext.kernelLayer.b.eval()

        W_ext_old = W_ext.copy()
        for (i, j) in matching:
            W_ext[i] = W_ext_old[j]

        W_ext -= W_ext[0, :]
        b_ext -= b_ext[0]

        model_loss = numpy.sum(numpy.abs(W_true - W_ext))
        model_loss += numpy.sum(numpy.abs(b_true - b_ext))
        """
        model_loss = 1

        return model_loss, repr_loss

    def compare(self, X_test, X_unif, scaler=None):
        if X_test is not None:
            y = self.query(X_test)
            p = self.query_probas(X_test)

        y_u = self.query(X_unif)
        p_u = self.query_probas(X_unif)

        for mtype in ['extr', 'base']:
            for mode in ['passive', 'adapt-local', 'adapt-oracle']:
                files = glob.glob('experiments/KLR/{}/models/{}_{}_[0-9]*.pkl'.
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

                    try:
                        l1_model, l1_repr = self.compare_models(model_ext,
                                                                samples,
                                                                scaler=scaler)
                    except IOError:
                        l1_model, l1_repr = numpy.nan

                    print '%s,%s,%d,%s,%.2e,%.2e,%.2e,%.2e,%.2e' % \
                          (self.dataset, mode, int(samples), mtype, 1-acc,
                           1-acc_u, l1, l1_u, l1_model)

if __name__ == '__main__':
    build_model()
