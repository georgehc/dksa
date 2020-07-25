"""
Code for loading survival analysis datasets for our benchmarking code

Author: George H. Chen (georgechen [at symbol] cmu.edu)
"""
#!/usr/bin/env python
import csv
import os
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines.datasets import load_regression_dataset
from pycox.datasets import metabric, support
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_dataset(dataset, random_seed_offset=0):
    """
    Loads a survival analysis dataset (supported public datasets: pbc, gbsg2,
    recid).

    Parameters
    ----------
    dataset : string
        One of 'toy', 'recid', 'metabric', 'rotterdam-gbsg2', or
        'support2_onehot'

    random_seed_offset : int, optional (default=0)
        Offset to add to random seed in shuffling the data.

    Returns
    -------
    X_train : 2D numpy array, shape = [n_samples, n_features]
        Training feature vectors.

    y_train : 2D numpy array, shape = [n_samples, 2]
        Survival labels (first column is for observed times, second column
        is for event indicators) for training data. The i-th row corresponds to
        the i-th row in `X_train`.

    X_test : 2D numpy array
        Test feature vectors. Same features as for training.

    y_test : 2D numpy array
        Test survival labels.

    feature_names : list
        List of strings specifying the names of the features (columns of
        `X_train` and `X_test`).

    compute_features_and_transformer : function
        Function for fitting and then transforming features into some
        "standardized"/"normalized" feature space. This should be applied to
        training feature vectors prior to using a learning algorithm (unless the
        learning algorithm does not need this sort of normalization). This
        function returns both the normalized features and a transformer object
        (see the next output for how to use this transformer object).

    transform_features : function
        Function that, given feature vectors (e.g., validation/test data) and a
        transformer object (created via `compute_features_and_transformer`),
        transforms the feature vectors into a normalized feature space.
    """
    if dataset == 'toy':
        regression_dataset = load_regression_dataset()
        regression_dataset_nparray = np.array(regression_dataset)
        X = regression_dataset_nparray[:, :3]
        y = regression_dataset_nparray[:, 3:]

        feature_names = [str(idx) for idx in range(X.shape[1])]

        def compute_features_and_transformer(features):
            scaler = StandardScaler()
            new_features = scaler.fit_transform(features)
            return new_features, scaler

        def transform_features(features, transformer):
            return transformer.transform(features)

        dataset_random_seed = 0

    elif dataset == 'recid':
        if not os.path.isfile('data/recid_X.txt') \
                or not os.path.isfile('data/recid_y.txt') \
                or not os.path.isfile('data/recid_feature_names.txt'):
            X = []
            y = []
            with open('data/recid.csv', 'r') as f:
                header = True
                for row in csv.reader(f):
                    if header:
                        feature_names = row[1:-4]
                        header = False
                    elif len(row) == 19:
                        black = float(row[1])  # indicator
                        alcohol = float(row[2])  # indicator
                        drugs = float(row[3])  # indicator
                        super_ = float(row[4])  # indicator
                        married = float(row[5])  # indicator
                        felon = float(row[6])  # indicator
                        workprg = float(row[7])  # indicator
                        property_ = float(row[8])  # indicator
                        person = float(row[9])  # indicator
                        priors = float(row[10])  # no. of prior convictions
                        educ = float(row[11])  # years of schooling
                        rules = float(row[12])  # no. of prison rule violations
                        age = float(row[13])  # in months
                        tserved = float(row[14])  # time served in months
                        time = float(row[16])
                        cens = 1. - float(row[17])
                        X.append((black, alcohol, drugs, super_, married,
                                  felon, workprg, property_, person, priors,
                                  educ, rules, age, tserved))
                        y.append((time, cens))
            X = np.array(X, dtype=np.float)
            y = np.array(y, dtype=np.float)

            with open('data/recid_feature_names.txt', 'w') as f:
                f.write("\n".join(feature_names))
            np.savetxt('data/recid_X.txt', X)
            np.savetxt('data/recid_y.txt', y)

        X = np.loadtxt('data/recid_X.txt')
        y = np.loadtxt('data/recid_y.txt')
        feature_names = [line.strip() for line
                         in open('data/recid_feature_names.txt').readlines()]

        def compute_features_and_transformer(features):
            new_features = np.zeros_like(features)
            transformer = StandardScaler()
            cols_standardize = [9, 10, 11, 12, 13]
            cols_leave = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            new_features[:, cols_standardize] = \
                transformer.fit_transform(features[:, cols_standardize])
            new_features[:, cols_leave] = features[:, cols_leave]
            return new_features, transformer

        def transform_features(features, transformer):
            new_features = np.zeros_like(features)
            cols_standardize = [9, 10, 11, 12, 13]
            cols_leave = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            new_features[:, cols_standardize] = \
                transformer.transform(features[:, cols_standardize])
            new_features[:, cols_leave] = features[:, cols_leave]
            return new_features

        dataset_random_seed = 3959156915

    elif dataset == 'metabric':
        df = metabric.read_df()
        X = df[['x0', 'x1', 'x2', 'x3', 'x4',
                'x5', 'x6', 'x7', 'x8']].to_numpy()
        y = df[['duration', 'event']].to_numpy()

        # for now, just use indices as feature names
        feature_names = [str(idx) for idx in range(9)]

        # possible actual feature names (taken from the DeepSurv paper but needs
        # verification; the ordering might be off)
        # feature_names = ['MKI67', 'EGFR', 'PGR', 'ERBB2',
        #                  'hormone treatment indicator',
        #                  'radiotherapy indicator',
        #                  'chemotherapy indicator',
        #                  'ER-positive indicator',
        #                  'age at diagnosis']

        def compute_features_and_transformer(features):
            new_features = np.zeros_like(features)
            transformer = StandardScaler()
            cols_standardize = [0, 1, 2, 3, 8]
            cols_leave = [4, 5, 6, 7]
            new_features[:, cols_standardize] = \
                transformer.fit_transform(features[:, cols_standardize])
            new_features[:, cols_leave] = features[:, cols_leave]
            return new_features, transformer

        def transform_features(features, transformer):
            new_features = np.zeros_like(features)
            cols_standardize = [0, 1, 2, 3, 8]
            cols_leave = [4, 5, 6, 7]
            new_features[:, cols_standardize] = \
                transformer.transform(features[:, cols_standardize])
            new_features[:, cols_leave] = features[:, cols_leave]
            return new_features

        dataset_random_seed = 1332972993

    elif dataset == 'support2_onehot':
        with open('data/support2.csv', 'r') as f:
            csv_reader = csv.reader(f)
            header = True
            X = []
            y = []
            for row in csv_reader:
                if header:
                    header = False
                else:
                    row = row[1:]

                    age = float(row[0])
                    sex = int(row[2] == 'female')

                    race = row[16]
                    if race == '':
                        race = 0
                    elif race == 'asian':
                        race = 1
                    elif race == 'black':
                        race = 2
                    elif race == 'hispanic':
                        race = 3
                    elif race == 'other':
                        race = 4
                    elif race == 'white':
                        race = 5

                    num_co = int(row[8])
                    diabetes = int(row[22])
                    dementia = int(row[23])

                    ca = row[24]
                    if ca == 'no':
                        ca = 0
                    elif ca == 'yes':
                        ca = 1
                    elif ca == 'metastatic':
                        ca = 2

                    meanbp = row[29]
                    if meanbp == '':
                        meanbp = np.nan
                    else:
                        meanbp = float(meanbp)

                    hrt = row[31]
                    if hrt == '':
                        hrt = np.nan
                    else:
                        hrt = float(hrt)

                    resp = row[32]
                    if resp == '':
                        resp = np.nan
                    else:
                        resp = float(resp)

                    temp = row[33]
                    if temp == '':
                        temp = np.nan
                    else:
                        temp = float(temp)

                    wblc = row[30]
                    if wblc == '':
                        wblc = np.nan
                    else:
                        wblc = float(wblc)

                    sod = row[38]
                    if sod == '':
                        sod = np.nan
                    else:
                        sod = float(sod)

                    crea = row[37]
                    if crea == '':
                        crea = np.nan
                    else:
                        crea = float(crea)

                    d_time = float(row[5])
                    death = int(row[1])

                    X.append([age, sex, race, num_co, diabetes, dementia, ca,
                              meanbp, hrt, resp, temp, wblc, sod, crea])
                    y.append([d_time, death])

        X = np.array(X)
        y = np.array(y)

        not_nan_mask = ~np.isnan(X).any(axis=1)
        X = X[not_nan_mask]
        y = y[not_nan_mask]

        feature_names = ['age', 'sex', 'num.co', 'diabetes', 'dementia', 'ca',
                         'meanbp', 'hrt', 'resp', 'temp', 'wblc', 'sod', 'crea',
                         'race_blank', 'race_asian', 'race_black',
                         'race_hispanic', 'race_other', 'race_white']

        categories = [list(range(int(X[:, 2].max()) + 1))]

        def compute_features_and_transformer(features):
            new_features = np.zeros((features.shape[0], 19))
            scaler = StandardScaler()
            encoder = OneHotEncoder(categories=categories)
            cols_standardize = [0, 7, 8, 9, 10, 11, 12, 13]
            cols_leave = [1, 4, 5]
            cols_categorical = [2]
            new_features[:, [0, 6, 7, 8, 9, 10, 11, 12]] = \
                scaler.fit_transform(features[:, cols_standardize])
            new_features[:, [1, 3, 4]] = features[:, cols_leave]
            new_features[:, 13:] = \
                encoder.fit_transform(features[:, cols_categorical]).toarray()
            new_features[:, 2] = features[:, 3] / 9.
            new_features[:, 5] = features[:, 6] / 2.
            transformer = (scaler, encoder)
            return new_features, transformer

        def transform_features(features, transformer):
            new_features = np.zeros((features.shape[0], 19))
            scaler, encoder = transformer
            cols_standardize = [0, 7, 8, 9, 10, 11, 12, 13]
            cols_leave = [1, 4, 5]
            cols_categorical = [2]
            new_features[:, [0, 6, 7, 8, 9, 10, 11, 12]] = \
                scaler.transform(features[:, cols_standardize])
            new_features[:, [1, 3, 4]] = features[:, cols_leave]
            new_features[:, 13:] = \
                encoder.transform(features[:, cols_categorical]).toarray()
            new_features[:, 2] = features[:, 3] / 9.
            new_features[:, 5] = features[:, 6] / 2.
            return new_features

        dataset_random_seed = 331231101

    elif dataset == 'rotterdam-gbsg2':
        # ----------------------------------------------------------------------
        # snippet of code from DeepSurv repository
        datasets = defaultdict(dict)
        with h5py.File('data/gbsg_cancer_train_test.h5', 'r') as fp:
            for ds in fp:
                for array in fp[ds]:
                    datasets[ds][array] = fp[ds][array][:]
        # ----------------------------------------------------------------------

        feature_names = ['horTh', 'tsize', 'menostat', 'age', 'pnodes',
                         'progrec', 'estrec']

        X_train = datasets['train']['x']
        y_train = np.array([datasets['train']['t'], datasets['train']['e']]).T
        X_test = datasets['test']['x']
        y_test = np.array([datasets['test']['t'], datasets['test']['e']]).T

        def compute_features_and_transformer(features):
            new_features = np.zeros_like(features)
            transformer = StandardScaler()
            cols_standardize = [3, 4, 5, 6]
            cols_leave = [0, 2]
            new_features[:, cols_standardize] = \
                transformer.fit_transform(features[:, cols_standardize])
            new_features[:, cols_leave] = features[:, cols_leave]
            new_features[:, 1] = features[:, 1] / 2.
            return new_features, transformer

        def transform_features(features, transformer):
            new_features = np.zeros_like(features)
            cols_standardize = [3, 4, 5, 6]
            cols_leave = [0, 2]
            new_features[:, cols_standardize] = \
                transformer.transform(features[:, cols_standardize])
            new_features[:, cols_leave] = features[:, cols_leave]
            new_features[:, 1] = features[:, 1] / 2.
            return new_features

        dataset_random_seed = 1831262265
        rng = np.random.RandomState(dataset_random_seed)
        shuffled_indices = rng.permutation(len(X_train))

        X_train = X_train[shuffled_indices]
        y_train = y_train[shuffled_indices]

    else:
        raise NotImplementedError('Unsupported dataset: %s' % dataset)

    if dataset != 'rotterdam-gbsg2' and dataset != 'deepsurv_nonlinear':
        rng = np.random.RandomState(dataset_random_seed + random_seed_offset)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.3, random_state=rng)

    return X_train, y_train, X_test, y_test, feature_names, \
            compute_features_and_transformer, transform_features
