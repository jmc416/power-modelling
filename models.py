from sklearn.tree import DecisionTreeClassifier

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

