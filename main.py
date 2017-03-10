import load
import models
import preprocessing
import visualisation


def classify_and_predict():

    X, y = load_model_data()
    test_X = load_test_data()

    model = models.fit_bagged_decision_tree(X, y)

    output_labels = model.predict(test_X)

    probabilities = model.predict_proba(test_X)

    data_rows, _ = load_test_rows(False, False, False)

    output = []
    for i, row in enumerate(data_rows):
        output.append((row['id'], probabilities[i][0], output_labels[i]))

    sorted_scores = sorted(output, key=lambda r: r[2])
    with open('output_scores', 'w') as f:
        f.writelines([str(r) + '\n' for r in sorted_scores])

    return output_labels, probabilities


def load_model_data():
    """

    """
    data_rows, features, label_rows = load_training_rows(True, True, True)

    X, y = preprocessing.labelled_training_data(data_rows, label_rows, features, load.LABEL_NAME)

    return X, y


def load_test_data():
    """

    """
    training_rows, features, _ = load_training_rows(True, True, True)
    data_rows, _ = load_test_rows(True, True, True)

    X = preprocessing.test_data(data_rows, features, training_rows)

    return X


def load_test_rows(transform_dates=True, transform_categorical_features=False,
                   add_timeseries_features=True):
    """

    """
    data_rows = load.load_test_data()
    historical_data = load.load_historical_test_data()
    features = load.load_features()
    timeseries_features = load.TIMESERIES_FEATURES

    data_rows, features = transformations(add_timeseries_features, data_rows, features,
                                          historical_data, timeseries_features,
                                          transform_categorical_features, transform_dates)

    return data_rows, features


def load_training_rows(transform_dates=True, transform_categorical_features=False,
                       add_timeseries_features=True):
    """

    """
    data_rows = load.load_training_data()
    historical_data = load.load_historical_training_data()
    features = load.load_features()
    timeseries_features = load.TIMESERIES_FEATURES

    data_rows, features = transformations(add_timeseries_features, data_rows, features,
                                          historical_data, timeseries_features,
                                          transform_categorical_features, transform_dates)

    label_rows = load.load_training_labels()

    return data_rows, features, label_rows


def transformations(add_timeseries_features, data_rows, features, historical_data,
                    timeseries_features, transform_categorical_features, transform_dates):
    """Transform the data"""
    if transform_dates:
        data_rows = preprocessing.transform_dates(data_rows, features)

    if transform_categorical_features:
        data_rows, categorical_value_maps = preprocessing.transform_categorical_features(data_rows,
                                                                                         features)

    if add_timeseries_features:
        timeseries_rows = preprocessing.extract_timeseries_rows(historical_data, features,
                                                                timeseries_features)
        data_rows, features = preprocessing.add_timeseries_features(data_rows, timeseries_rows,
                                                                    features, timeseries_features)
    return data_rows, features


def plot_all_features(plot_categorical=True, plot_continuous=True,
                      data_rows=None, features=None, label_rows=None):
    """

    """
    def should_log_x(feature):
        log_x = bool(int(feature['log_x']))
        is_date = bool(int(feature['is_date']))
        return log_x and not is_date

    if not all([data_rows, features, label_rows]):
        data_rows, features, label_rows = load_training_rows(transform_dates=True,
                                                             transform_categorical_features=False,
                                                             add_timeseries_features=True)

    for feature_name, feature in features.iteritems():

        if feature_name in load.TIMESERIES_FEATURES:
            continue

        print 'Plotting ', feature_name
        print feature

        if int(feature['is_categorical']) and plot_categorical:
            visualisation.categorical_plot(feature_name, data_rows, label_rows)

        elif plot_continuous:
            visualisation.continuous_plot(feature_name, data_rows, label_rows,
                                          log_x=should_log_x(feature),
                                          bandwidth=float(feature['bandwidth']) or 0.2,
                                          is_date=bool(int(feature['is_date'])))

