import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.manifold import TSNE

import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X, y = load_svmlight_file('../targets/circle/test.scale')
    X = X.todense()

    plt.scatter(X[:,0].A1, X[:,1].A1, 20, c=y)
    plt.show()
