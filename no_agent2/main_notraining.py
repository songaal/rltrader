import logging
import os
import math
import pandas as pd
import numpy as np
from no_agent2 import data_manager, settings
from no_agent2.data_manager import seq2dataset
from no_agent2.policy_learner import PolicyLearner

if __name__ == '__main__':
    exchange = 'binance'
    symbol = 'btc_usdt'
    periods = '4h'
    load_file_name = 'action_%s_%s_%s.csv' % (exchange, symbol, periods)
    model_name = 'model_67.hdf5'
    model_ver = '20180927172502'
    model_path = os.path.join(settings.PROJECT_DIR, 'models/%s/%s/%s/%s' % (exchange, symbol, periods, model_ver))

    # 코인 데이터 준비
    chart_data = data_manager.load_chart_data(os.path.join(settings.PROJECT_DIR, 'data/ingest_data/', load_file_name))
    chart_data['date'] = pd.to_datetime(chart_data['date'])
    prep_data = data_manager.preprocess(chart_data)
    chart_data = data_manager.build_training_data(prep_data)

    # 기간 필터링
    x_in = chart_data[(chart_data['date'] >= '2018-07-01') & (chart_data['date'] < '2018-09-18')]
    x_in = x_in.dropna()

    # 학습 데이터 분리
    features_training_data = [
        'open_close_ratio', 'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio',
        'close_ma10_ratio', 'volume_ma10_ratio',
        'close_ma20_ratio', 'volume_ma20_ratio',
        'close_ma60_ratio', 'volume_ma60_ratio',
        'close_ma120_ratio', 'volume_ma120_ratio'
    ]
    x, t, date, close = seq2dataset(x_in, window_size=5, features_training_data=features_training_data)
# y:-0.33   w: 0.10
    # 강화학습 시작
    policy_learner = PolicyLearner(symbol=symbol, x_train=x, lr=.001)


    for i in range(len(x)):
        y = policy_learner.trade(x[i].reshape(1, 5, 15), model_path='%s/%s' % (model_path, model_name))

        is_match = (t[i] > 0 and y > 0) or (t[i] == 0 and y == 0) or (t[i] < 0 and y < 0)
        diff_rate = (t[i][0] - y) / t[i][0]

        print('날짜: {} 가격: {} \t정답: {} \t예측: {} \t매치여부: {} 차이 비율: {}'.format(pd.to_datetime(date[i]),
                                                                                       close[i],
                                                                                       math.ceil(t[i][0] * 100) / 100,
                                                                                       math.ceil(y * 100) / 100,
                                                                                       is_match,
                                                                                       diff_rate))
