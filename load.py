import csv

FEATURES_FILE = 'features.csv'

TEST_DATA_FILE = 'test_data.csv'
TRAINING_DATA_FILE = 'train_data.csv'
TRAINING_LABELS_FILE = 'train_output.csv'
TRAINING_HISTORICAL_DATA_FILE = 'hist_data.csv'
TEST_HISTORICAL_DATA_FILE = 'test_hist_data.csv'

LABEL_NAME = 'churned'

TIMESERIES_FEATURES = ['price_p1_var', 'price_p2_var', 'price_p3_var',
                       'price_p1_fix', 'price_p2_fix', 'price_p3_fix']


def extract_rows(file_path):
    """Read data from a csv file.

    :param str file_path: The path to the file
    :rtype: list[dict[str, str]]
    """
    with open(file_path) as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def load_test_data():
    return extract_rows(TEST_DATA_FILE)


def load_training_data():
    return extract_rows(TRAINING_DATA_FILE)


def load_historical_training_data():
    return extract_rows(TRAINING_HISTORICAL_DATA_FILE)


def load_historical_test_data():
    return extract_rows(TEST_HISTORICAL_DATA_FILE)


def load_features():
    return {row['name']: row for row in extract_rows(FEATURES_FILE)}


def load_training_labels():
    """Read in the training labels

    :rtype: list[dict[str, int]]
    """
    ids = [row['id'] for row in extract_rows(TRAINING_DATA_FILE)]
    with open(TRAINING_LABELS_FILE) as f:
        labels = [int(r.strip()) for r in f.readlines()]
    return [{'id': _id, LABEL_NAME: label} for _id, label in zip(ids, labels)]
