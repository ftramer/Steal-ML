import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


def make_plots_lr(filenames, title="Dummy Title"):
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
    row_names = [r'$L_{test}$', r'$L_{unif}$',
                 r'probas $L_{test}$', r'probas $L_{unif}$']

    methods = [
        ('base', 'passive'),
        ('base', 'adapt-local'),
        ('base', 'adapt-oracle'),
        ('base', 'lowd-meek'),
        ('extr', 'passive')]
    method_names = ['Agnostic Learning',
                    'Active Learning',
                    'Boundary Finding',
                    'Lowd - Meek',
                    'Equation Solving (with conf. scores)']

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, axs = plt.subplots(nrows=len(rows), ncols=1, sharex=True,
                            figsize=(25, 15))

    for ax, row in zip(axs, row_names):
        ax.set_ylabel(row, size='large')

    for row, target in enumerate(rows):
            ax = axs[row]
            ax.yaxis.grid(True)
            for (mode, method), m_name in zip(methods, method_names):

                if mode in datas.values()[0]['mode'].values \
                        and method in datas.values()[0]['method'].values:

                    ys = []
                    for data in datas.values():
                        xy = data[(data['mode'] == mode) &
                                  (data['method'] == method)].sort('budget')

                        if 'binary' in data['method'].values and mode == 'extr':
                            xy = data[(data['mode'] == mode) &
                                      (data['method'] == 'binary')]

                        y = xy[target].values

                        if len(y) < len(x):
                            y = np.hstack(([1]*(len(x)-len(y)), y))

                        if len(y) > len(x):
                            y = y[-len(x):]

                        if 'loss' in target:
                            y = np.maximum(y, 1e-6)

                        y = np.hstack((y[0],
                                       [min(y[:i+1]) for i in range(1, len(y))]))

                        ys.append(y)

                    mean_y = np.mean(np.array(ys), axis=0)
                    err_y = np.std(np.array(ys), axis=0)

                    ax.errorbar(x, mean_y, err_y, label=m_name)

                    #if 'accuracy' in target:
                    ax.set_yscale('log')
                    ax.set_ylim(0, None, auto=True)
                    ax.set_xlim(0, 105)

            if row == 0:
                ax.legend(loc='upper right')
                ax.set_title(title)
            if row == len(rows) - 1:
                ax.set_xlabel(r'Budget ($\alpha \cdot \texttt{num\_unknowns}$)')

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('title', type=str, help='figure title')
    parser.add_argument('filenames', nargs='+', type=str,
                        help='a list of filenames')

    args = parser.parse_args()

    title = args.title
    filenames = args.filenames

    make_plots_lr(filenames, title=title)

if __name__ == "__main__":
    main()
