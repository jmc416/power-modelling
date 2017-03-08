import unittest

import preprocessing


class PreprocessingTest(unittest.TestCase):

    def test_simple_categorical_features(self):
        features = {'type': {'is_categorical': True}}

        rows = [
            {'type': 'small', 'weight': 100},
            {'type': 'medium', 'weight': 120},
            {'type': 'big', 'weight': 150},
            {'type': 'big', 'weight': 190},
        ]

        expected_rows = [
            {'type': 1, 'weight': 100},
            {'type': 2, 'weight': 120},
            {'type': 3, 'weight': 150},
            {'type': 3, 'weight': 190},
        ]

        expected_value_map = {
            'type': {
                1: 'small',
                2: 'medium',
                3: 'big'
            }
        }
        transformed_rows, value_map = preprocessing.transform_categorical_features(rows, features)

        self.assertEqual(expected_rows, transformed_rows)
        self.assertEqual(expected_value_map, value_map)

    def test_missing_categorical_features(self):
        features = {'type': {'is_categorical': True}}

        rows = [
            {'type': 'small', 'weight': 100},
            {'type': 'medium', 'weight': 120},
            {'type': 'big', 'weight': 150},
            {'type': '', 'weight': 190},
        ]

        expected_rows = [
            {'type': 1, 'weight': 100},
            {'type': 2, 'weight': 120},
            {'type': 3, 'weight': 150},
            {'type': 4, 'weight': 190},
        ]

        transformed_rows, _ = preprocessing.transform_categorical_features(rows, features)

        self.assertEqual(expected_rows, transformed_rows)

    def test_simple_dates(self):

        features = {'date': {'is_date': True},
                    'type': {'is_date': False}}

        rows = [
            {'type': 'big', 'date': '2016-10-02'},
            {'type': 'big', 'date': '2016-10-01'},
        ]

        expected_rows = [
            {'type': 'big', 'date': 1475362800.0},
            {'type': 'big', 'date': 1475276400.0},
        ]
        transformed_rows = preprocessing.transform_dates(rows, features)

        self.assertEqual(expected_rows, transformed_rows)

    def test_missing_dates(self):

        features = {'date': {'is_date': True},
                    'type': {'is_date': False}}

        rows = [
            {'type': 'big', 'date': ''},
            {'type': 'big', 'date': '2016-10-01'},
        ]

        expected_rows = [
            {'type': 'big', 'date': 0},
            {'type': 'big', 'date': 1475276400.0},
        ]
        transformed_rows = preprocessing.transform_dates(rows, features)

        self.assertEqual(expected_rows, transformed_rows)

    def test_vectorise(self):
        features = {'type': {'is_categorical': True},
                    'weight': {'is_categorical': False}}

        rows = [
            {'type': 1, 'weight': 100},
            {'type': 2, 'weight': 120},
            {'type': 3, 'weight': 150},
            {'type': 3, 'weight': 190},
        ]

        expected_array = preprocessing.np.array([
            [1, 100],
            [2, 120],
            [3, 150],
            [3, 190]
        ])

        array = preprocessing.vectorise(rows, features)
        self.assert_array_elements_equal(array, expected_array)

    def test_encoding(self):
        features = {'type': {'is_categorical': True},
                    'weight': {'is_categorical': False}}

        data = preprocessing.np.array([
            [1, 100],
            [2, 120],
            [3, 150],
            [3, 190]
        ])

        expected_array = preprocessing.np.array([
            [1, 0, 0, 100],
            [0, 1, 0, 120],
            [0, 0, 1, 150],
            [0, 0, 1, 190],
        ])

        array = preprocessing.encode_categorical_features(data, features).toarray()
        self.assert_array_elements_equal(array, expected_array)

    def test_labelled_data_rows(self):

        features = {'type': {'is_categorical': True, 'is_date': False},
                    'date': {'is_date': True, 'is_categorical': False},
                    'weight': {'is_categorical': False, 'is_date': False}}

        rows = [
            {'type': 'small', 'weight': 100, 'date': '2016-10-02', 'id': 1},
            {'type': 'medium', 'weight': 120, 'date': '2016-10-02', 'id': 2},
            {'type': 'big', 'weight': 150, 'date': '2016-10-02', 'id': 3},
            {'type': 'big', 'weight': 190, 'date': '2016-10-02', 'id': 4},
        ]

        label_rows = [
            {'id': 1, 'churned': False},
            {'id': 2, 'churned': False},
            {'id': 3, 'churned': True},
            {'id': 4, 'churned': True},
        ]

        expected_data = preprocessing.np.array([
            [1, 0, 0, 1475362800.0, 100],
            [0, 1, 0, 1475362800.0, 120],
            [0, 0, 1, 1475362800.0, 150],
            [0, 0, 1, 1475362800.0, 190],
        ])
        expected_labels = [0, 0, 1, 1]

        data, labels = preprocessing.labelled_training_data(rows, label_rows, features, 'churned')
        data_array = data.toarray()
        self.assert_array_elements_equal(expected_data, data_array)
        self.assertItemsEqual(expected_labels, labels)

    def test_real_data(self):
        features = {'activity_new': {'is_categorical': '1', 'is_date': '0'},
                    'campaign_disc_ele': {'is_categorical': '1', 'is_date': '0'},
                    'channel_sales': {'is_categorical': '1', 'is_date': '0'},
                    'churned': {'is_categorical': '1', 'is_date': '0'},
                    'cons_12m': {'is_categorical': '0', 'is_date': '0'},
                    'cons_gas_12m': {'is_categorical': '0', 'is_date': '0'},
                    'cons_last_month': {'is_categorical': '0', 'is_date': '0'},
                    'date_activ': {'is_categorical': '0', 'is_date': '1'},
                    'date_end': {'is_categorical': '0', 'is_date': '1'},
                    'date_first_activ': {'is_categorical': '0', 'is_date': '1'},
                    'date_modif_prod': {'is_categorical': '0', 'is_date': '1'},
                    'date_renewal': {'is_categorical': '0', 'is_date': '1'},
                    'forecast_base_bill_ele': {'is_categorical': '0', 'is_date': '0'},
                    'forecast_base_bill_year': {'is_categorical': '0', 'is_date': '0'},
                    'forecast_bill_12m': {'is_categorical': '0', 'is_date': '0'},
                    'forecast_cons': {'is_categorical': '0', 'is_date': '0'},
                    'forecast_cons_12m': {'is_categorical': '0', 'is_date': '0'},
                    'forecast_cons_year': {'is_categorical': '0', 'is_date': '0'},
                    'forecast_discount_energy': {'is_categorical': '0', 'is_date': '0'},
                    'forecast_meter_rent_12m': {'is_categorical': '0', 'is_date': '0'},
                    'forecast_price_energy_p1': {'is_categorical': '0', 'is_date': '0'},
                    'forecast_price_energy_p2': {'is_categorical': '0', 'is_date': '0'},
                    'forecast_price_pow_p1': {'is_categorical': '0', 'is_date': '0'},
                    'has_gas': {'is_categorical': '1', 'is_date': '0'},
                    'id': {'is_categorical': '1', 'is_date': '0'},
                    'imp_cons': {'is_categorical': '0', 'is_date': '0'},
                    'margin_gross_pow_ele': {'is_categorical': '0', 'is_date': '0'},
                    'margin_net_pow_ele': {'is_categorical': '0', 'is_date': '0'},
                    'nb_prod_act': {'is_categorical': '0', 'is_date': '0'},
                    'net_margin': {'is_categorical': '0', 'is_date': '0'},
                    'num_years_antig': {'is_categorical': '0', 'is_date': '0'},
                    'origin_up': {'is_categorical': '1', 'is_date': '0'},
                    'pow_max': {'is_categorical': '0', 'is_date': '0'},
                    'price_date': {'is_categorical': '0', 'is_date': '1'},
                    'price_p1_fix': {'is_categorical': '0', 'is_date': '0'},
                    'price_p1_var': {'is_categorical': '0', 'is_date': '0'},
                    'price_p2_fix': {'is_categorical': '0', 'is_date': '0'},
                    'price_p2_var': {'is_categorical': '0', 'is_date': '0'},
                    'price_p3_fix': {'is_categorical': '0', 'is_date': '0'},
                    'price_p3_var': {'is_categorical': '0', 'is_date': '0'}
                    }
        rows = [
            {'forecast_base_bill_ele': '',
             'date_end': '2017-01-03',
             'forecast_cons_12m': '2634.23',
             'activity_new': 'apdekpcbwosbxepsfxclislboipuxpop',
             'origin_up': 'lxidpiddsbxsbosboudacockeimpuepw',
             'forecast_meter_rent_12m': '16.42',
             'campaign_disc_ele': '',
             'id': 'cf81de72ff7997ed10729751059cf7a3',
             'cons_last_month': '12091',
             'nb_prod_act': '1',
             'date_modif_prod': '2011-01-03',
             'forecast_base_bill_year': '',
             'forecast_cons': '',
             'margin_gross_pow_ele': '27.0',
             'forecast_discount_energy': '0.0',
             'imp_cons': '93.12',
             'pow_max': '11.950999999999999',
             'has_gas': 'f',
             'date_activ': '2011-01-03',
             'forecast_bill_12m': '',
             'date_renewal': '2016-01-04',
             'cons_12m': '186838',
             'date_first_activ': '',
             'channel_sales': 'foosdfpfkusacimwkcsosbicdxkicaua',
             'cons_gas_12m': '0',
             'forecast_price_pow_p1': '44.31137796',
             'num_years_antig': '5',
             'margin_net_pow_ele': '27.0',
             'forecast_price_energy_p2': '0.086163',
             'forecast_price_energy_p1': '0.16405799999999998',
             'net_margin': '210.18',
             'forecast_cons_year': '738'}
        ]
        label_rows = [
            {'id': 'cf81de72ff7997ed10729751059cf7a3', 'churned': 1}
        ]

        data, labels = preprocessing.labelled_training_data(rows, label_rows, features, 'churned')

    def assert_array_elements_equal(self, array, expected_array):
        for i, row in enumerate(array):
            for j, value in enumerate(row):
                self.assertEqual(value, expected_array[i][j])

    def test_parse_date(self):
        string = '2015-01-02'
        self.assertEqual(preprocessing.parse_date(string), 1420156800.0)

    def test_extract_timeseries(self):

        timeseries_features = {'price_1': None, 'price_2': None}

        timeseries_rows = [
            {'id': '1', 'price_date': '2015-01-01', 'price_1': 10, 'price_2': 1},
            {'id': '1', 'price_date': '2015-02-01', 'price_1': 20, 'price_2': 2},
            {'id': '2', 'price_date': '2015-01-01', 'price_1': 30, 'price_2': 3},
            {'id': '2', 'price_date': '2015-02-01', 'price_1': 20, 'price_2': 2},
            {'id': '2', 'price_date': '2015-03-01', 'price_1': 10, 'price_2': 1},
        ]

        expected_rows = [
            {'id': '1',
             'price_1': {1420070400.0: 10, 1422748800.0: 20},
             'price_2': {1420070400.0: 1, 1422748800.0: 2}},
            {'id': '2',
             'price_1': {1420070400.0: 30, 1422748800.0: 20, 1425168000.0: 10},
             'price_2': {1420070400.0: 3, 1422748800.0: 2, 1425168000.0: 1}}
        ]

        rows = preprocessing.extract_timeseries_rows(timeseries_rows, {}, timeseries_features)
        self.assertItemsEqual([str(r) for r in expected_rows],
                              [str(r) for r in rows])


if __name__ == '__main__':
    unittest.main()