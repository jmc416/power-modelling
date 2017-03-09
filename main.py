import load
import preprocessing
import visualisation


def load_training_rows(transform_dates=True, transform_categorical_features=False,
                       add_timeseries_features=True):

    data_rows = load.load_training_data()
    historical_data = load.load_historical_training_data()
    features = load.load_features()
    timeseries_features = load.TIMESERIES_FEATURES

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

    label_rows = load.load_training_labels()

    return data_rows, features, label_rows


def plot_all_features(plot_categorical=True, plot_continuous=True,
                      data_rows=None, features=None, label_rows=None):

    if not all([data_rows, features, label_rows]):
        data_rows, features, label_rows = load_training_rows(transform_dates=True,
                                                             transform_categorical_features=False,
                                                             add_timeseries_features=True)

    for feature_name, feature in features.iteritems():

        if feature_name in load.TIMESERIES_FEATURES:
            continue

        print 'Plotting ', feature_name

        try:
            if int(feature['is_categorical']) and plot_categorical:
                visualisation.categorical_plot(feature_name, data_rows, label_rows)

            elif plot_continuous:
                visualisation.continuous_plot(feature_name, data_rows, label_rows,
                                              log_x=not bool(int(feature['is_date'])))
        except Exception as e:
            e
