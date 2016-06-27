import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.manifold import TSNE

import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X, y = load_svmlight_file('../targets/breast-cancer/test.scale')
    X = X.todense()

    model = TSNE(n_components=2, verbose=1)
    Y = model.fit_transform(X)
    plt.scatter(Y[:,0], Y[:,1], 20, c=y)
    plt.show()
