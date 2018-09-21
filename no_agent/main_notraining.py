import logging
import os

import pandas as pd
import numpy as np
from no_agent import data_manager, settings
from no_agent.policy_learner import PolicyLearner

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
        dataset_Y.append([subset.weight.values[window_size]])
        date.append([subset.date.values[window_size]])
        close.append(subset.close.values[window_size])
    return np.array(dataset_X), np.array(dataset_Y), np.array(date), np.array(close)


if __name__ == '__main__':
    exchange = 'binance'
    symbol = 'btc_usdt'
    periods = '4h'
    load_file_name = 'weight_%s_%s_%s.csv' % (exchange, symbol, periods)
    model_name = 'model_442.hdf5'
    model_ver = '20180921094333'
    model_path = os.path.join(settings.PROJECT_DIR, 'models/%s/%s/%s/%s' % (exchange, symbol, periods, model_ver))

    # 코인 데이터 준비
    chart_data = data_manager.load_chart_data(os.path.join(settings.PROJECT_DIR, 'data/ingest_data/', load_file_name))
    chart_data['date'] = pd.to_datetime(chart_data['date'])
    prep_data = data_manager.preprocess(chart_data)
    chart_data = data_manager.build_training_data(prep_data)

    # 기간 필터링
    x_in = chart_data[(chart_data['date'] >= '2018-08-01') & (chart_data['date'] < '2018-09-18')]
    x_in = x_in.dropna()

    # 학습 데이터 분리
    features_training_data = [
        'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio',
        'close_ma10_ratio', 'volume_ma10_ratio',
        'close_ma20_ratio', 'volume_ma20_ratio',
        'close_ma60_ratio', 'volume_ma60_ratio',
        'close_ma120_ratio', 'volume_ma120_ratio'
    ]
    x, y, date, close = seq2dataset(x_in, window_size=5, features_training_data=features_training_data)

    # 강화학습 시작
    policy_learner = PolicyLearner(symbol=symbol, x_train=x, lr=.001)

    y_results = []
    for i in range(len(x)):
        weight = policy_learner.trade(x[i].reshape(1, 5, 15), model_path='%s/%s' % (model_path, model_name))
        y_results.append(weight)
        print('순서: {}, 날짜: {}, 가격: {}, 정답: {}, \t예측: {}'.format(i, date[i], close[i], y[i][0], weight))
