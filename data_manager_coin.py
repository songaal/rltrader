import pandas as pd
import numpy as np
import talib

def load_chart_data(fpath):
    chart_data = pd.read_csv(fpath, thousands=',', header=None)
    chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
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
    training_data['rsi14'] = np.zeros(len(training_data))
    training_data['rsi14'] = talib.RSI(prep_data['close'], timeperiod=14)

    training_data['sma5'] = np.zeros(len(training_data))
    training_data['sma5'] = talib.SMA(prep_data['close'], 5)

    training_data['sma20'] = np.zeros(len(training_data))
    training_data['sma20'] = talib.SMA(prep_data['close'], 20)

    training_data['sma120'] = np.zeros(len(training_data))
    training_data['sma120'] = talib.SMA(prep_data['close'], 120)

    training_data['ema12'] = np.zeros(len(training_data))
    training_data['ema12'] = talib.SMA(prep_data['close'], 12)

    training_data['ema26'] = np.zeros(len(training_data))
    training_data['ema26'] = talib.SMA(prep_data['close'], 26)

    upper, middle, lower = talib.BBANDS(prep_data['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    training_data['dn'] = np.zeros(len(training_data))
    training_data['mavg'] = np.zeros(len(training_data))
    training_data['up'] = np.zeros(len(training_data))
    training_data['pctB'] = np.zeros(len(training_data))

    training_data['dn'] = lower
    training_data['mavg'] = middle
    training_data['up'] = upper
    training_data['pctB'] = (prep_data['close'] - lower) / (upper - lower)

    macd, macdsignal, macdhist = talib.MACD(prep_data['close'], 12, 26, 9)
    training_data['macd'] = np.zeros(len(training_data))
    training_data['signal'] = np.zeros(len(training_data))
    training_data['macd'] = macd
    training_data['macdsignal'] = macdsignal

    training_data['open_lastclose_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'open_lastclose_ratio'] = \
        (training_data['open'][1:].values - training_data['close'][:-1].values) / \
        training_data['close'][:-1].values
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
        training_data['volume'][:-1] \
            .replace(to_replace=0, method='ffill') \
            .replace(to_replace=0, method='bfill').values

    # windows = [5, 10, 20, 60, 120]
    # for window in windows:
    #     training_data['close_ma%d_ratio' % window] = \
    #         (training_data['close'] - training_data['close_ma%d' % window]) / \
    #         training_data['close_ma%d' % window]
    #     training_data['volume_ma%d_ratio' % window] = \
    #         (training_data['volume'] - training_data['volume_ma%d' % window]) / \
    #         training_data['volume_ma%d' % window]

    return training_data


# chart_data = pd.read_csv(fpath, encoding='CP949', thousands=',', engine='python')
