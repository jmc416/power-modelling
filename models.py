from sklearn.tree import DecisionTreeClassifier

import load
import preprocessing


def load_data():

    training_rows = load.load_training_data()[:10]
    training_labels = load.load_training_labels()[:10]
    features = load.load_features()

    X, y = preprocessing.labelled_training_data(training_labels, training_rows, features,
                                                load.LABEL_NAME)
    return X, y


def fit_decision_tree(X, y):

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    return clf

