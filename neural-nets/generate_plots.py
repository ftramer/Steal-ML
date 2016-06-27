import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


def make_plots_lr(filenames):
    datas = {}
    for idx, filename in enumerate(filenames):
        datas[idx] = pd.read_csv(filename, header=0)

    rows = ['loss', 'loss_u', 'probas', 'probas_u']
    row_names = [r'$L_{test}$', r'$L_{unif}$',
                 r'probas $L_{test}$', r'probas $L_{unif}$']

    methods = [
        ('base', 'passive'),
        ('base', 'adapt-local'),
        ('base', 'adapt-oracle'),
        #('extr', 'passive')
    ]
    method_names = [
        'Agnostic Learning',
        'Active Learning',
        'Boundary Finding',
        'Equation Solving (with conf. scores)'
    ]

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
                ys = []

                for data in datas.values():
                    xy = data[(data['mode'] == mode) &
                              (data['method'] == method)].sort('budget')
                    x = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

                    y = xy[target].values

                    if 'loss' in target:
                        y = np.maximum(y, 1e-6)

                    y = np.hstack((y[0],
                                   [min(y[:i+1]) for i in range(1, len(y))]))
                    ys.append(y)

                mean_y = np.mean(np.array(ys), axis=0)
                err_y = np.std(np.array(ys), axis=0)

                ax.errorbar(x, mean_y, err_y, label=m_name)
                ax.set_yscale('log')
                ax.set_ylim(0, None, auto=True)
                ax.set_xlim(0, 120)
            if row == 0:
                ax.set_title('Extraction of Neural Networks')
                ax.legend(loc='upper right')
            if row == len(rows) - 1:
                ax.set_xlabel(r'Budget ($\alpha \cdot \texttt{num\_unknowns}$)')

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+', type=str,
                        help='a list of filenames')

    args = parser.parse_args()

    filenames = args.filenames

    make_plots_lr(filenames)

if __name__ == "__main__":
    main()
