import os

from no_agent import settings, data_manager
from no_agent.policy_learner import PolicyLearner

if __name__ == '__main__':
    exchange = 'binance'
    symbol = 'btc_usdt'
    periods = '4h'

    timestr = settings.get_time_str()

    # 코인 데이터 준비
    chart_data = data_manager.load_chart_data(os.path.join(settings.BASE_DIR, 'weight_%s_%s_%s.csv' % (exchange, symbol, periods)))

    prep_data = data_manager.preprocess(chart_data)
    training_data = data_manager.build_training_data(prep_data)

    # 기간 필터링
    training_data = training_data[(training_data['date'] >= '2017-01-01') &
                                  (training_data['date'] <= '2017-12-31')]
    training_data = training_data.dropna()

    # 차트 데이터 분리
    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume', 'weight']
    chart_data = training_data[features_chart_data]

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
    training_data = training_data[features_training_data]

    # 강화학습 시작
    policy_learner = PolicyLearner(symbol=symbol, chart_data=chart_data, training_data=training_data, lr=.001)

    policy_learner.fit(balance=10000000, num_epoches=1000, discount_factor=0, start_epsilon=.5)

    # 정책 신경망을 파일로 저장
    model_path = os.path.join(settings.BASE_DIR, 'model_%s_%s_%s_%s.h5' % exchange, symbol, periods, timestr)
    policy_learner.policy_network.save_model(model_path)
