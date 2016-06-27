import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def plotcontour(predict, X, y):

    # create a mesh to plot in
    xx, yy = np.meshgrid(np.arange(-1, 1, .02),
                         np.arange(-1, 1, .02))

    # title for the plots
    titles = 'SVC with RBF kernel'

    mesh = np.c_[xx.ravel(), yy.ravel()]
    print mesh.size


    Z = predict(np.c_[xx.ravel(), yy.ravel()])
    if not hasattr(Z, 'reshape'):
        Z = np.array(Z)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles)

    plt.show()


if __name__ == '__main__':
    # import some data to play with
    X, y = datasets.make_circles(n_samples=800, noise=0.07, factor=0.4)
    C = 1.0  # SVM regularization parameter
    clf = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
    plotcontour(clf.predict, X, y)
