import pandas as pd
import numpy as np
import argparse


def aggregate(filenames, out_dir):
    datas = {}
    x = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    for idx, filename in enumerate(filenames):
        if '/LR/' in filename:
            skip = 2
        else:
            skip = 0
        try:
            datas[idx] = pd.read_csv(filename, skiprows=skip, header=0)
        except ValueError:
            print "Failed to load %s" % filename

    rows = ['loss', 'loss_u', 'probas', 'probas_u']

    methods = {'base': ['passive', 'adapt-local', 'adapt-oracle', 'lowd-meek'],
               'extr': ['passive', 'binary']}

    for mode in methods.keys():

        res_mode = pd.DataFrame(
                columns=['Q_by_U', 'best L_test_bar', 'best L_unif_bar',
                         'mean_budget']
            )
        res_mode['Q_by_U'] = x
        res_mode = res_mode.replace(np.nan, np.inf)

        for method in methods[mode]:
            if method not in datas.values()[0]['method'].values:
                print 'skipping {}-{}'.format(mode, method)
                continue

            res = pd.DataFrame(
                columns=['Q_by_U', 'L_test_bar', 'L_test_std', 'L_unif_bar',
                         'L_unif_std', 'L_test_TV_bar', 'L_test_TV_std',
                         'L_unif_TV_bar', 'L_unif_TV_std']
            )
            res['Q_by_U'] = x

            for (idx, target) in enumerate(rows):

                budgets = []
                ys = []

                for data in datas.values():
                    xy = data[(data['mode'] == mode) &
                              (data['method'] == method)].sort('budget')
                    y = xy[target].values

                    y = np.hstack((y[0], [min(y[:i+1]) for i in range(1, len(y))]))
                    ys.append(y)

                    budget = xy['budget']
                    budgets.append(np.array(budget))

                mean_y = np.mean(np.array(ys), axis=0)
                err_y = np.std(np.array(ys), axis=0)
                mean_x = np.mean(np.array(budgets), axis=0)

                res[res.columns[2*idx+1]] = mean_y
                res[res.columns[2*idx+2]] = err_y

            output = out_dir + '/' + mode + '_' + method + '.dat'
            res.to_csv(output, index=False, sep=',')

            res_mode['mean_budget'] = mean_x

            for col_idx in range(1, 3):
                res_mode[res_mode.columns[col_idx]] \
                    = np.minimum(res_mode[res_mode.columns[col_idx]],
                                 res[res.columns[2*col_idx-1]])

        for col_idx in range(1, 3):
            res_mode[res_mode.columns[col_idx]] \
                = (1-res_mode[res_mode.columns[col_idx]])*100

        output = out_dir + '/' + mode + '_summary.dat'
        res_mode.to_csv(output, index=False, float_format='%.6f', sep=',')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str, help='output directory')
    parser.add_argument('filenames', nargs='+', type=str,
                        help='a list of filenames')
    args = parser.parse_args()

    out_dir = args.out_dir
    filenames = args.filenames

    aggregate(filenames, out_dir)

if __name__ == "__main__":
    main()
