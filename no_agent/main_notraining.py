import logging
import os

import pandas as pd

import settings
from no_agent import data_manager
from no_agent.policy_learner import PolicyLearner

if __name__ == '__main__':
    exchange = 'binance'
    symbol = 'btc_usdt'
    periods = '4h'
    model_ver = '20180911124438'

    # 로그 기록
    log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % symbol)
    timestr = settings.get_time_str()
    file_handler = logging.FileHandler(filename=os.path.join(
        log_dir, "%s_%s.log" % (symbol, timestr)), encoding='utf-8')
    stream_handler = logging.StreamHandler()
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
        handlers=[file_handler, stream_handler], level=logging.DEBUG)

    # 데이터 준비
    chart_data = data_manager.load_chart_data(
        os.path.join(settings.BASE_DIR, 'weight_%s_%s_%s.csv' % (exchange, symbol, periods)))
    chart_data['date'] = pd.to_datetime(chart_data['date'])
    prep_data = data_manager.preprocess(chart_data)
    chart_data = data_manager.build_training_data(prep_data)

    # 기간 필터링
    x = chart_data[(chart_data['date'] >= '2018-08-01') & (chart_data['date'] < '2018-09-18')]
    x = x.dropna()

    y = x['weight']
    del x['weight']

    # 차트 데이터 분리
    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
    chart_data = x[features_chart_data]

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
    training_data = x[features_training_data]

    # 비 학습 투자 시뮬레이션 시작
    policy_learner = PolicyLearner(symbol=symbol, chart_data=chart_data, training_data=training_data)
    model_path = os.path.join(settings.BASE_DIR, 'model_%s_%s_%s_%s.h5' % exchange, symbol, periods, timestr)
    policy_learner.trade(model_path=model_path)
