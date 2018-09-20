import os

import pandas as pd

from no_agent import settings, data_manager
from no_agent.policy_learner import PolicyLearner

if __name__ == '__main__':
    exchange = 'binance'
    symbol = 'btc_usdt'
    periods = '4h'
    timestr = settings.get_time_str()

    # 코인 데이터 준비
    chart_data = data_manager.load_chart_data(os.path.join(settings.BASE_DIR, 'weight_%s_%s_%s.csv' % (exchange, symbol, periods)))
    chart_data['date'] = pd.to_datetime(chart_data['date'])
    prep_data = data_manager.preprocess(chart_data)
    chart_data = data_manager.build_training_data(prep_data)

    # 기간 필터링
    x_train = chart_data[(chart_data['date'] >= '2017-11-01') & (chart_data['date'] < '2018-07-01')]
    x_train = x_train.dropna()
    x_test = chart_data[(chart_data['date'] > '2018-07-01') & (chart_data['date'] < '2018-08-01')]
    x_test = x_test.dropna()

    # x, y 데이터 분리
    y_train = x_train['weight']
    del x_train['weight']
    y_test = x_test['weight']
    del x_test['weight']

    # 차트 데이터 분리
    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
    chart_data = chart_data[features_chart_data]

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
    training_data = x_train[features_training_data]
    test_data = x_test[features_training_data]

    # 강화학습 시작
    policy_learner = PolicyLearner(symbol=symbol, chart_data=chart_data, training_data=training_data, test_data=test_data, lr=.001)

    policy_learner.fit(x_train=training_data, y_train=y_train, x_test=test_data, y_test=y_test, num_epoches=1000)

    # 정책 신경망을 파일로 저장
    model_path = os.path.join(settings.BASE_DIR, 'model_%s_%s_%s_%s.h5' % exchange, symbol, periods, timestr)
    policy_learner.policy_network.save_model(model_path)
