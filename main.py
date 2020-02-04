import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets

from KNearestNeighboursModel import KNearestNeighboursModel
from KNearestNeighboursModel import print_ndarray


def create_plot(train_X, train_y, test_X, test_y, test_prob):
    fig, axes = plt.subplots()
    axes.set_title('Classification using K Nearest Neighbours')
    axes.xaxis.set_label_text('Feature 1')
    axes.yaxis.set_label_text('Feature 2')
    axes.grid()

    # Train
    axes.scatter(train_X[:, 0], train_X[:, 1],
            marker='.', c=train_y, cmap='Set1')
    # Test
    axes.scatter(test_X[:, 0], test_X[:, 1],
            marker='x', c=test_y, cmap='Set1', s=(100*test_prob**5))
    # Label each test point with it's probablity

    # TODO: Legend

    fig.savefig('result.svg')
    plt.show()


def generate_train_data(n_feats, n_clusters, n_samples_per_cluster):
    base = np.arange(n_samples_per_cluster * n_feats).reshape(
            n_samples_per_cluster, n_feats)

    cluster_samples = ([np.concatenate(
            (x * 10 + base, x * np.ones((n_samples_per_cluster, 1)) + 1),
            axis=1)
        for x in range(n_clusters)])

    train_data = np.vstack(cluster_samples)
    np.random.shuffle(train_data)
    print_ndarray(train_data, 'train_data')

    train_X, train_y = train_data[:, :-1], train_data[:, -1]
    return train_X, train_y


def generate_train_data_iris():
    iris = datasets.load_iris()
    return iris['data'][:, 2:4], iris['target']


def generate_test_data():
    '''
    # Generate one test sample for each class
    points = np.ones((1, n_feats))
    classes = np.arange(n_clusters).reshape(n_clusters, 1)
    test_X = points + 10 * classes
    '''
    test_X = np.array([
        [1, 4],
        [15, 11],
        [30, 34],
        [9, 1],
        [12, 1],
        [33, 44]])
    return test_X


def generate_test_data_iris():
    test_X = np.array([
        [1.5, 0.2],
        [4, 1.5],
        [6, 2.3],
        [5, 1.8],
        [2.8, 0.6]])
    return test_X


def main():
    # Generate Train Data
    n_feats = 2
    n_clusters=3
    n_samples_per_cluster=3
    train_X, train_y = generate_train_data_iris()

    # Fit
    k_neighbours = 10
    model = KNearestNeighboursModel(k_neighbours)
    model.fit(train_X, train_y)


    # Generate Test Data
    test_X = generate_test_data_iris()
    print_ndarray(test_X, 'test_X')

    # Predict
    test_y = model.predict(test_X)
    print_ndarray(test_y, 'test_y')

    # Predict probablities
    test_prob = model.predict_proba(test_X)
    print_ndarray(test_prob, 'test_prob')


    create_plot(train_X, train_y, test_X, test_y, test_prob)


if __name__ == '__main__':
    main()

