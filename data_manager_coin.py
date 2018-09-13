import pandas as pd
import numpy as np
import talib


def load_chart_data(fpath):
    chart_data = pd.read_csv(fpath, thousands=',', header=None)
    chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    return chart_data


def preprocess(chart_data):
    prep_data = chart_data
    open = prep_data['open']
    high = prep_data['high']
    low = prep_data['low']
    close = prep_data['close']
    volume = prep_data['volume']


    prep_data['rsi14'] = np.zeros(len(chart_data))
    prep_data['rsi14'] = talib.RSI(close, timeperiod=14)

    prep_data['stoch_9_6_slowk'] = np.zeros(len(chart_data))
    prep_data['stoch_9_6_slowd'] = np.zeros(len(chart_data))
    prep_data['stoch_9_6_slowk'], prep_data['stoch_9_6_slowd'] = talib.STOCH(high, low, close, fastk_period=9, slowk_period=6, slowk_matype=0, slowd_period=6, slowd_matype=0)

    prep_data['stoch_14_slowk'] = np.zeros(len(chart_data))
    prep_data['stoch_14_slowd'] = np.zeros(len(chart_data))
    prep_data['stoch_14_slowk'], prep_data['stoch_14_slowd'] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)

    prep_data['macd'] = np.zeros(len(chart_data))
    prep_data['macdsignal'] = np.zeros(len(chart_data))
    prep_data['macd'], prep_data['macdsignal'], macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    prep_data['adx'] = np.zeros(len(chart_data))
    prep_data['adx'] = talib.ADX(high, low, close, timeperiod=14)

    prep_data['willr'] = np.zeros(len(chart_data))
    prep_data['willr'] = talib.WILLR(high, low, close, timeperiod=14)

    prep_data['cci'] = np.zeros(len(chart_data))
    prep_data['cci'] = talib.CCI(high, low, close, timeperiod=14)

    prep_data['ultosc'] = np.zeros(len(chart_data))
    prep_data['ultosc'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

    prep_data['roc'] = np.zeros(len(chart_data))
    prep_data['roc'] = talib.ROC(close, timeperiod=10)

    for window in [5, 10, 20, 50, 100, 200]:
        prep_data['close_ma{}'.format(window)] = np.zeros(len(chart_data))
        prep_data['close_ma{}'.format(window)] = talib.MA(prep_data['close'], timeperiod=window)
        prep_data['volume_ma{}'.format(window)] = np.zeros(len(chart_data))
        prep_data['volume_ma{}'.format(window)] = talib.MA(prep_data['close'], timeperiod=window)

    return prep_data


def build_training_data(prep_data):
    training_data = prep_data
    open = prep_data['open']
    high = prep_data['high']
    low = prep_data['low']
    close = prep_data['close']
    volume = prep_data['volume']


    for window in [5, 10, 20, 50, 100, 200]:
        training_data['close_ma{}'.format(window)] = np.zeros(len(training_data))
        training_data['close_ma{}'.format(window)] = talib.MA(close, timeperiod=window)
        training_data['volume_ma{}'.format(window)] = np.zeros(len(training_data))
        training_data['volume_ma{}'.format(window)] = talib.MA(close, timeperiod=window)

    return training_data
