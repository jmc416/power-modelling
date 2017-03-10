import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB

import load
import preprocessing


def load_data_portion(denominator=0, offset=0):
    """Load a portion of the data.

    :param denominator: The number of partitions to divide the data into e.g. 3 (thirds)
    :param offset: The partition to load e.g. 2 (the second third)
    :return:
    """
    training_rows = load.load_training_data()
    training_labels = load.load_training_labels()
    features = load.load_features()

    number_of_rows = min(len(training_rows), len(training_labels))
    portion_length = (number_of_rows / denominator) if denominator else number_of_rows
    slice_start = offset * portion_length
    slice_end = slice_start + portion_length
    print 'Returning %s samples' % portion_length

    X, y = preprocessing.labelled_training_data(
        training_rows[slice_start:slice_end], training_labels[slice_start:slice_end],
        features, load.LABEL_NAME)
    return X, y


def fit_decision_tree(X, y):

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    return clf


def fit_bagged_decision_tree(X, y):
    clf = BaggingClassifier()
    clf.fit(X, y)
    return clf


def fit_forest(X, y):
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    forest.fit(X, y)
    return forest


def fit_naive_bayes(X, y):
    gnb = GaussianNB()
    gnb.fit(X, y)
    return gnb


def feature_importances(X, y):

    forest = fit_forest(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
