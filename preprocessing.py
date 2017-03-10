from __future__ import division

from datetime import date
from itertools import chain
from itertools import groupby
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
    print 'Transforming categorical features'
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


def parse_date(value, date_format=DATE_FORMAT):
    return time.mktime(time.strptime(value, date_format))


def transform_dates(rows, features, date_format=DATE_FORMAT):
    """Return a new list of rows with the dates transformed into epoch seconds

    :param list[dict[str, Any]] rows: a list of data
    :param dict[str, dict[str, bool] features:
    :param str date_format: the format of the date for the time.strptime parser
    :rtype: tuple[list[dict[str, Any]]
    """
    print 'Transforming dates'
    def transform(feature, value):
        if feature not in features:
            raise ValueError('Unknown feature')
        elif not int(features[feature]['is_date']):
            return value
        elif not value:
            return EMPTY_DATE_POLICY
        else:
            return parse_date(value, date_format)

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

    for support vector machine

    :param list[dict[str, Any]] rows: a list of data
    :param dict[str, dict[str, bool] features:
    :param str date_format: the format of the date for the time.strptime parser
    :rtype: list[dict[str, Any]
    """
    pass


def timeseries_max(_, y):
    return max(y) if y else 0


def timeseries_min(_, y):
    return min(y) if y else 0


def timeseries_range(x, y):
    return timeseries_max(x, y) - timeseries_min(x, y) if y else 0


def timeseries_sum_returns(_, y):
    """A simple measure of the direction of change of the timeseries"""
    if y:
        ret = [b - a for b, a in zip(y[1:], y[:-1])]
        return sum(ret)
    else:
        return 0


def extract_timeseries_rows(timeseries_rows, features, timeseries_features):
    """Extract the timeseries

    :param list[dict[str, Any]] timeseries_rows: a list of data
    :param dict[str, dict[str, bool] features:
    :param dict[str, dict[str, bool] timeseries_features:
    :return The rows with extra features added, along with the extra features' details
    :rtype tuple(list[dict[str, Any], list[list[str]])
    """
    print 'Extracting timeseries rows'
    output = []
    for id, id_rows in groupby(timeseries_rows, key=lambda row: row['id']):
        output_row = {'id': id}
        id_rows = list(id_rows)
        for feature_name in timeseries_features:
            timeseries = {parse_date(id_row['price_date']): id_row[feature_name]
                          for id_row in id_rows}
            output_row[feature_name] = timeseries
        output.append(output_row)
    return output


def make_xy(feature_name, row):
    """Extract the timestamps and values (x, y) from a timeseries row"""
    try:
        x, y = zip(*filter(lambda tup: tup[1] > 0.0,
                           sorted([(k, float(v or 0)) for k, v in row[feature_name].iteritems()],
                                  key=lambda tup: tup[0]))
                   )
    except ValueError:
        x, y = [], []
    return x, y


def add_timeseries_features(rows, timeseries_rows, features,
                            timeseries_features, new_feature_names=None):
    """Extract some features from timeseries features and add them to the features set

    :param list[dict[str, Any]] rows: a list of data
    :param dict[str, dict[str, bool] features:
    :param dict[str, dict[str, bool] timeseries_features:
    :return The rows with extra features added, along with the extra features' details
    :rtype tuple(list[dict[str, Any], list[list[str]])
    """
    print 'Adding timeseries features'
    derived_features = {
        'max': {'name': 'max', 'is_date': 0, 'is_categorical': 0,
                'function': timeseries_max},
        'min': {'name': 'min', 'is_date': 0, 'is_categorical': 0,
                'function': timeseries_min},
        'range': {'name': 'range', 'is_date': 0, 'is_categorical': 0,
                  'function': timeseries_range},
        'sum_returns': {'name': 'sum_returns', 'is_date': 0,
                        'is_categorical': 0, 'function': timeseries_sum_returns},
    }
    if new_feature_names:
        derived_features = {k: derived_features[k]
                            for k in new_feature_names if k in derived_features}

    new_feature_template = {'is_date': 0, 'is_categorical': 0, 'log_x': False, 'bandwidth': 0.2}
    new_features = {}

    timeseries_rows_index = {row['id']: i for i, row in enumerate(timeseries_rows)}

    for i, row in enumerate(rows):
        try:
            timeseries_row = timeseries_rows[timeseries_rows_index[row['id']]]
            assert timeseries_row['id'] == row['id']
            for timeseries_name, timeseries in timeseries_row.iteritems():
                if timeseries_name in timeseries_features:
                    for feature_name in derived_features:
                        x, y = make_xy(timeseries_name, timeseries_row)
                        derived_value = derived_features[feature_name]['function'](x, y)
                        new_feature_name = timeseries_name + '_' + feature_name
                        row[new_feature_name] = derived_value
                        new_features[new_feature_name] = new_feature_template
        except Exception as e:
            print row
            print e
            pass

    # Add timeseries features to features
    features = dict(chain(features.items(), new_features.items()))
    return rows, features


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

