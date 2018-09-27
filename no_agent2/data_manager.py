import pandas as pd
import numpy as np


def load_chart_data(fpath):
    chart_data = pd.read_csv(fpath, thousands=',', header=None)
    chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'action_B', 'action_H', 'action_S']
    return chart_data


def preprocess(chart_data):
    prep_data = chart_data
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        prep_data['close_ma{}'.format(window)] = prep_data['close'].rolling(window).mean()
        prep_data['volume_ma{}'.format(window)] = (
            prep_data['volume'].rolling(window).mean())
    return prep_data


def build_training_data(prep_data):
    training_data = prep_data

    training_data['open_close_ratio'] = np.zeros(len(training_data))
    training_data['open_close_ratio'] = \
        (training_data['open'].values - training_data['close'].values) / \
        training_data['open'].values

    training_data['high_close_ratio'] = \
        (training_data['high'].values - training_data['close'].values) / \
        training_data['close'].values
    training_data['low_close_ratio'] = \
        (training_data['low'].values - training_data['close'].values) / \
        training_data['close'].values
    training_data['close_lastclose_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'close_lastclose_ratio'] = \
        (training_data['close'][1:].values - training_data['close'][:-1].values) / \
        training_data['close'][:-1].values
    training_data['volume_lastvolume_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'volume_lastvolume_ratio'] = \
        (training_data['volume'][1:].values - training_data['volume'][:-1].values) / \
        training_data['volume'][:-1]\
            .replace(to_replace=0, method='ffill') \
            .replace(to_replace=0, method='bfill').values

    windows = [5, 10, 20, 60, 120]
    for window in windows:
        training_data['close_ma%d_ratio' % window] = \
            (training_data['close'] - training_data['close_ma%d' % window]) / \
            training_data['close_ma%d' % window]
        training_data['volume_ma%d_ratio' % window] = \
            (training_data['volume'] - training_data['volume_ma%d' % window]) / \
            training_data['volume_ma%d' % window]

    return training_data


# def seq2dataset(seq, window_size, features_training_data):
#     dataset_I = []
#     dataset_X = []
#     dataset_Y = []
#
#     for i in range(len(seq) - window_size):
#         subset = seq[i:(i + window_size + 1)]
#
#         for si in range(len(subset) - 1):
#             features = subset[features_training_data].values[si]
#             dataset_I.append(features)
#         dataset_X.append(dataset_I)
#         dataset_I = []
#         dataset_Y.append([subset.action_B.values[window_size], subset.action_H.values[window_size], subset.action_S.values[window_size]])
#
#     return np.array(dataset_X), np.array(dataset_Y)

def seq2dataset(seq, window_size, features_training_data):
    dataset_I = []
    dataset_X = []
    dataset_Y = []
    date = []
    close = []

    for i in range(len(seq) - window_size):
        subset = seq[i:(i + window_size + 1)]
        for si in range(len(subset) - 1):
            features = subset[features_training_data].values[si]
            dataset_I.append(features)

        dataset_X.append(dataset_I)
        dataset_I = []
        dataset_Y.append([subset.action_B.values[window_size],
                          subset.action_H.values[window_size],
                          subset.action_S.values[window_size]])
        date.append(subset.date.values[window_size])
        close.append(subset.close.values[window_size])
    return np.array(dataset_X), np.array(dataset_Y), np.array(date), np.array(close)

# chart_data = pd.read_csv(fpath, encoding='CP949', thousands=',', engine='python')
