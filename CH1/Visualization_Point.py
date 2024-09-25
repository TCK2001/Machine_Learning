import matplotlib.pyplot as plt

def visualization_base(X, y):
    plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker = 'o', label = 'versicolor')

    plt.xlabel('sepal length[cm]')
    plt.ylabel('petal length[cm]')
    plt.xlim(X[:50, 0].min() - 1, X[50:100, 0].max() + 1)
    plt.ylim(X[0:50, 1].min() - 1, X[50:100, 1].max() + 1)
    plt.legend(loc = 'upper left')
    plt.show()

