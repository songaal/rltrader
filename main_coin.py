import logging
import os
import settings
import data_manager_coin
import pandas
from policy_learner_coin import PolicyLearner
import datetime

# SELECT first("open") AS "first_open", max("high") AS "max_high", min("low") AS "min_low", last("close") AS "last_close", sum("volume") AS "sum_volume" FROM "coin_v2"."autogen"."binance_btc_usdt" GROUP BY time(1d) FILL(null)

if __name__ == '__main__':
    symbol = 'BTCUSDT'
    t_start = '2018-08-16'
    t_end = '2018-09-01'
    epoches = 1000
    balance = 100000

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
    training_data = training_data[(training_data['date'] >= t_start) &
                                  (training_data['date'] <= t_end)]
    training_data = training_data.dropna()

    # 차트 데이터 분리
    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
    chart_data = training_data[features_chart_data]

    # 학습 데이터 분리
    features_training_data = [
        'rsi14', 'sma5', 'sma10', 'sma20', 'sma120', 'obv', 'ad',
        'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'close_ma10_ratio', 'close_ma20_ratio', 'close_ma120_ratio',
        'volume_ma5_ratio', 'volume_ma10_ratio', 'volume_ma20_ratio', 'volume_ma120_ratio'
    ]

    training_data = training_data[features_training_data]
    training_start = datetime.datetime.now()
    # 강화학습 시작
    policy_learner = PolicyLearner(
        symbol=symbol, chart_data=chart_data, training_data=training_data,
        min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=.0000000005, lr=.00000001)
    policy_learner.fit(balance=balance, num_epoches=epoches, discount_factor=0, start_epsilon=.5)

    training_end = datetime.datetime.now()
    delta = training_end - training_start
    logging.info("학습 소요시간: %s", delta)
    logging.info("모델파일: models/%s/model_%s.h5", symbol, timestr)
    # 정책 신경망을 파일로 저장
    model_dir = os.path.join(settings.BASE_DIR, 'models/%s' % symbol)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model_%s.h5' % timestr)
    policy_learner.policy_network.save_model(model_path)
