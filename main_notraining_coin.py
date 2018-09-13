import logging
import os

import pandas
import sys

import settings
import data_manager_coin
from policy_learner_coin import PolicyLearner


if __name__ == '__main__':
    symbol = 'XRPBTC'
    model_ver = '20180913145305'
    start = '2018-01-01'
    end = '2018-09-12'

    # 로그 기록
    log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % symbol)
    timestr = settings.get_time_str()
    if not os.path.exists('logs/%s' % symbol):
        os.makedirs('logs/%s' % symbol)
    file_handler = logging.FileHandler(filename=os.path.join(
        log_dir, "%s_%s.log" % (symbol, timestr)), encoding='utf-8')
    stream_handler = logging.StreamHandler()
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
        handlers=[file_handler, stream_handler], level=logging.DEBUG)

    # 주식 데이터 준비
    chart_data = data_manager_coin.load_chart_data(
        os.path.join(settings.BASE_DIR,
                     'data/chart_data/{}.csv'.format(symbol)))
    chart_data['date'] = pandas.to_datetime(chart_data['date'])
    prep_data = data_manager_coin.preprocess(chart_data)
    training_data = data_manager_coin.build_training_data(prep_data)

    # 기간 필터링
    training_data = training_data[(training_data['date'] >= start) &
                                  (training_data['date'] <= end)]
    training_data = training_data.dropna()

    # 차트 데이터 분리
    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
    chart_data = training_data[features_chart_data]

    # 학습 데이터 분리
    features_training_data = [
        'rsi14', 'stoch_9_6_slowk', 'stoch_9_6_slowd', 'stoch_14_slowk', 'stoch_14_slowd', 'macd', 'macdsignal', 'adx', 'willr', 'cci', 'ultosc', 'roc',
        'close_ma5', 'close_ma10', 'close_ma20', 'close_ma50', 'close_ma100', 'close_ma200',
        'volume_ma5', 'volume_ma10', 'volume_ma20', 'volume_ma50', 'volume_ma100', 'volume_ma200'
    ]

    training_data = training_data[features_training_data]

    # 비 학습 투자 시뮬레이션 시작
    policy_learner = PolicyLearner(
        symbol=symbol, chart_data=chart_data, training_data=training_data,
        min_trading_unit=1, max_trading_unit=3)
    policy_learner.trade(balance=100000,
                         model_path=os.path.join(
                             settings.BASE_DIR,
                             'models/model_{}.h5'.format(model_ver)))
