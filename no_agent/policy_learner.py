import os
import locale
import logging
import numpy as np
from keras.datasets import mnist

from no_agent.policy_network import PolicyNetwork

logger = logging.getLogger(__name__)
locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')


class PolicyLearner:

    def __init__(self, symbol, chart_data, training_data=None, test_data=None, lr=0.01):
        self.symbol = symbol
        self.chart_data = chart_data

        self.training_data = training_data  # 학습 데이터
        self.test_data = test_data
        # 정책 신경망; 입력 크기 = 학습 데이터의 크기 + 에이전트 상태 크기
        self.num_features = self.training_data.shape
        self.policy_network = PolicyNetwork(input_dim=self.num_features, lr=lr)

    def fit(self, x_train, y_train, x_test, y_test, num_epoches=1000, batch_size=10):
        self.policy_network.fit(x_train=x_train, y_train=y_train,
                                epochs=num_epoches, batch_size=batch_size,
                                x_test=x_test, y_test=y_test)

    def trade(self, x, model_path=None):
        if model_path is None:
            return
        self.policy_network.load_model(model_path=model_path)
        # self.fit(learning=False)
        self.policy_network.predict(x)
