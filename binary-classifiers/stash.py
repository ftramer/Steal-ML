__author__ = 'Fan'


## collection of backup functions

def solve_poly_in_f(self):
    n_pts = int(2 * len(self.fmap([0] * self.n_features)))
    print 'requires %d pts' % n_pts
    pts_near_b_in_x, pts_near_b_in_x_label = self.collect_pts(n_pts)
    print 'done collecting points'
    print self.q
    m = len(pts_near_b_in_x)
    para = np.matrix(map(self.fmap, pts_near_b_in_x))

    print 'done solving'

    bb = -1 * np.ones(m).T
    self.w, _, _, _ = np.linalg.lstsq(para, bb)
    self.b = 1

    s_test = 0.0
    yy = [self.predict(d) for d in self.pts_near_b]
    for y1, y2 in zip(yy, self.pts_near_b_labels):
        if y1 == y2:
            s_test += 1.0
        else:
            print 'wants %d gets %d' % (y2, y1)

    score = s_test / float(len(yy))
    print 'score on training is %f' % max(score, 1 - score)

    if score < 0.5:
        self.w *= -1
        self.b *= -1
        return 1 - score
    else:
        return score


def train_SGD_for_poly_in_F(self):
    n_pts = int(2 * len(self.fmap([0] * self.n_features)))
    X, Y = self.collect_pts(n_pts)
    X_in_f = np.matrix([self.fmap(xx) for xx in X])
    clf = SGDClassifier()
    clf.fit(X_in_f, Y)

    self.clf2 = clf
    return clf.score(X_in_f, Y)


## poly kernel should find something useful here

def benchmark(self):
    # test
    error_clf = 0.0
    error_lrn = 0.0
    for x, y in zip(self.Xt.tolist(), self.Yt.tolist()):
        t = self.predict(x)
        logger.info('want %d, gets %d' % (y, t))
        if t != y:
            error_lrn += 1
        if self.query(x) != y:
            error_clf += 1

    pe_clf = 1 - error_clf / len(self.Yt)
    pe_lrn = 1 - error_lrn / len(self.Yt)

    self.real_score = pe_clf
    self.score = pe_lrn

    logger.info('real score is %f while learned score is %f' % (pe_clf, pe_lrn))


def predict(self, x):
    logger.debug('predicting %s', str(x))
    if self.kernel == 'rbf':
        if self.method == OfflineMethods.RT_in_F:
            xx = self.fmap.transform(x)
            return self.clf2.predict(xx)
        elif self.method == OfflineMethods.SLV_in_F:
            xx = self.fmap.transform(x)
            yy = np.inner(xx, self.w)
            b = self.b * np.ones(yy.shape)
            d = np.sign(np.inner(xx, self.w) + b)
            d[d == 1] = self.POS
            d[d == -1] = self.NEG
            return d
        else:
            assert False, 'unknown method'
    elif self.kernel == 'poly':
        xx = self.fmap(x)
        if self.method == OfflineMethods.SLV_in_F:
            d = np.inner(xx, self.w) + self.b
            return self.POS if d > 0 else self.NEG
        elif self.method == OfflineMethods.RT_in_F:
            return self.clf2.predict(xx)

# def solve_in_f_with_grid():
#     print '--------------- solve in F with grid -----------------'
#     bf_pts, bf_test, bn_q = [], [], []
#     gamma_score = []
#     for log2n in (5,):
#         # for log2n in log2n_range:
#         n_pts = 2**log2n
#         ex = RBFExtractor(test_data, rbf_svc.predict, Xt, Yt,
#                      n_features, M.SLV_in_F,
#                      error=1e-9, kernel='rbf',
#                      fmap=None, ftype=feature_type)
#         score_list, (s_train, gamma) = ex.grid_solve_in_f(n_pts)
#         y_pred = [ex.predict(x) for x in Xt]
#         s_test = sm.accuracy_score(Yt, y_pred)
#         q = ex.get_n_query()
#
#         print 'RBF DIM      : %d' % n_pts
#         print 'SCORE TRAIN  : %f' % s_train
#         print 'SCORE TEST   : %f' % s_test
#         print 'GAMMA        : %f' % gamma
#         print 'Q            : %d' % q
#         print ''
#         bf_pts.append(n_pts)
#         bf_test.append(s_test)
#         bn_q.append(q)
#
#         p = {
#             'legend': '%d dim' % n_pts,
#             'plot': score_list,
#         }
#         gamma_score.append(p)
#     with open(test_data + '-grid-gamma-SViF.pickle', 'wb') as f:
#         pickle.dump(gamma_score, f)
#
#     b.add('solve in F with grid',
#           bf_pts, 'dim. of RBF kernel',
#           (bf_test,), ('test',),
#           (bn_q,), ('number of queries',))
#     del bf_pts, bf_test, bn_q
