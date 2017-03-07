from __future__ import division

from datetime import date
import time

import numpy as np
from sklearn.preprocessing import OneHotEncoder

EMPTY_DATE_POLICY = 0
EMPTY_DATUM_POLICY = 0

DATE_FORMAT = '%Y-%m-%d'


def transform_categorical_features(rows, features):
    """Return a new list of rows with the categorical features transformed into integers, along with
    a mapping from the integers to the values

    :param list[dict[str, Any]] rows: a list of data
    :param dict[str, dict[str, bool] features:
    :rtype: tuple[list[dict[str, Any]], dict[int, str]]
    """
    categorical_features = {feature_name: {}
                            for feature_name, feature in features.iteritems()
                            if int(feature['is_categorical'])}

    def transform(feature, value):
        if feature not in categorical_features:
            return value
        elif value in categorical_features[feature]:
            return categorical_features[feature][value]
        else:
            categorical_features[feature][value] = len(categorical_features[feature].keys()) + 1
            return np.float64(categorical_features[feature][value])

    output = []
    for row in rows:
        transformed_row = {}
        for feature, value in row.iteritems():
            transformed_row[feature] = transform(feature, value)
        output.append(transformed_row)

    value_maps = {feature: {transformed_value: original_value
                            for original_value, transformed_value in values.iteritems()}
                  for feature, values in categorical_features.iteritems()}

    return output, value_maps


def transform_dates(rows, features, date_format=DATE_FORMAT):
    """Return a new list of rows with the dates transformed into epoch seconds

    :param list[dict[str, Any]] rows: a list of data
    :param dict[str, dict[str, bool] features:
    :param str date_format: the format of the date for the time.strptime parser
    :rtype: tuple[list[dict[str, Any]]
    """
    def transform(feature, value):
        if feature not in features:
            raise ValueError('Unknown feature')
        elif not int(features[feature]['is_date']):
            return value
        elif not value:
            return EMPTY_DATE_POLICY
        else:
            return time.mktime(time.strptime(value, date_format))

    output = []
    for row in rows:
        transformed_row = {}
        for feature, value in row.iteritems():
            transformed_row[feature] = transform(feature, value)
        output.append(transformed_row)

    return output


def format_timestamp(timestamp):
    """Turn an epoch timestamp into a readable string

    >>>format_timestamp(1475276400.0)
    '2016-10-01'

    :rtype: str
    """
    return date.fromtimestamp(timestamp).isoformat()


def normalise_features(rows, features):
    """Return a new list of rows with numerical features normalised to mean = 0 and std = 1

    :param list[dict[str, Any]] rows: a list of data
    :param dict[str, dict[str, bool] features:
    :param str date_format: the format of the date for the time.strptime parser
    :rtype: list[dict[str, Any]
    """
    pass


def timeseries_max(timeseries):
    return max(timeseries)


def timeseries_min(timeseries):
    return min(timeseries)


def timeseries_range(timeseries):
    return timeseries_max(timeseries) - timeseries_min(timeseries)


def add_timeseries_features(rows, features, timeseries_features):
    """Extract some features from timeseries features and add them to the features set

    :param list[dict[str, Any]] rows: a list of data
    :param dict[str, dict[str, bool] features:
    :param list[list[str]] timeseries_features:
    :return The rows with extra features added, along with the extra features' details
    :rtype tuple(list[dict[str, Any], list[list[str]])
    """
    derived_features = {
        'timeseries_max': {'name': 'timeseries_max', 'is_date': 0, 'is_categorical': 0,
                           'function': timeseries_max},
        'timeseries_min': {'name': 'timeseries_min', 'is_date': 0, 'is_categorical': 0,
                           'function': timeseries_min},
        'timeseries_range': {'name': 'timeseries_range', 'is_date': 0, 'is_categorical': 0,
                             'function': timeseries_range},
    }

    return rows


def vectorise(rows, features):
    """Return a numpy array of the rows

    :param list[dict[str, float]] rows:
    :param dict[str, dict[str, bool]] features:
    :rtype: np.array[np.float64]
    """
    features = sorted(set(features.keys()).intersection(set(rows[0].keys())))
    X = np.zeros([len(rows), len(features)])
    for i, row in enumerate(rows):
        for j, feature in enumerate(features):
            value = row.get(feature)
            if value == '':
                value = EMPTY_DATUM_POLICY
            X[i, j] = np.float64(value)
    return X


def encode_categorical_features(data, features):
    """Return a sparse one-hot encoding of the categorical features.

    Note that the categorical features will appear to the left of the output array

    :param np.array[np.float64] data:
    :param dict[str, dict[str, bool] features:
    :rtype: np.array[np.float64]
    """
    feature_names = sorted(features.keys())
    categorical_features = [True if int(features[feature]['is_categorical']) else False
                            for feature in feature_names]
    enc = OneHotEncoder(categorical_features=categorical_features)
    return enc.fit_transform(data)


def labelled_training_data(data_rows, label_rows, features, label_name):
    """Return processed and vectorised data and labels

    :param list[dict[str, Any]] data_rows:
    :param list[dict[str, Any]] label_rows:
    :param dict[str, dict[str, bool]] features:
    :param str label_name:
    :rtype: tuple[np.array, np.array]
    """
    data_by_id = {row['id']: row for row in data_rows}
    labels_by_id = {row['id']: row for row in label_rows}
    ids = list(set(data_by_id.keys()).intersection(set(labels_by_id.keys())))

    # Guarantee the labels are correct
    rows, labels = [], []
    for id in ids:
        row = data_by_id[id]
        if row.get('id'):
            row.pop('id')
        rows.append(row)
        labels.append(labels_by_id[id][label_name])

    rows, _ = transform_categorical_features(rows, features)
    rows = transform_dates(rows, features)

    features = {name: features[name] for name in rows[0].iterkeys()}
    data = vectorise(rows, features)

    X = encode_categorical_features(data, features)
    y = np.array([np.float64(1 if label else 0) for label in labels])
    return X, y

