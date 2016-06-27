__author__ = 'Fan'

import matplotlib.pyplot as plt
import numpy as np

font = {'family': 'serif',
        'serif': ['Palatino'],
        'size': 10}

plt.rc('font',**font)
plt.rc('text', usetex=True)


def drawy_yy(ax, x, base, ys, yys, xlabel, ylabel, yylabel, title,
             ys_std=None, yys_std=None):
    ax.grid()
    bh = ax.plot(x, base, 'g-', linewidth=2)

    yh = []
    y_legend = []
    for idx, (y, s, l) in enumerate(ys):
        if ys_std is not None and ys_std[idx] is not None:
            ax.errorbar(x, y, fmt=s, yerr=ys_std[idx], ms=5)
        yh += ax.plot(x, y, s, ms=5)
        y_legend.append(l)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, color='red')

    if base[0] > .95:
        ax.set_ylim([.5, 1.1])
    else:
        ax.set_ylim([.5, 1])

    ax2 = ax.twinx()
    ax2.set_ylabel(yylabel, color='blue')
    yyh = []
    yy_legend = []
    for idx, (yy, s, l) in enumerate(yys):
        if yys_std is not None and yys_std[idx] is not None:
            ax2.errorbar(x, yy, fmt=s, yerr=yys_std[idx], ms=5)

        yyh += ax2.plot(x, yy, s, ms=5)
        yy_legend.append(l)

    ax2.legend(bh + yh + yyh, ['original']+ y_legend + yy_legend, loc='lower right', fontsize=8)
    ax2.set_title(title)


def draw_heatmap(matrix, x_range, y_range, xlabel, ylabel, title, figname):
    from matplotlib.colors import Normalize

    class MidpointNormalize(Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    for i in range(0, len(x_range)):
        for j in range(0, len(y_range)):
            plt.text(i - .25, j, '%.3f' % matrix[j, i], size=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.xticks(np.arange(len(x_range)), x_range, rotation=45)
    plt.yticks(np.arange(len(y_range)), y_range, rotation=-45)
    plt.title(title)
    plt.savefig(figname)
