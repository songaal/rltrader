import os
import locale
import logging
import numpy as np

from no_agent.policy_network import PolicyNetwork

logger = logging.getLogger(__name__)
locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')


class PolicyLearner:

    def __init__(self, symbol, chart_data, training_data=None, lr=0.01):
        self.symbol = symbol
        self.chart_data = chart_data

        self.training_data = training_data  # 학습 데이터
        # 정책 신경망; 입력 크기 = 학습 데이터의 크기 + 에이전트 상태 크기
        self.num_features = self.training_data.shape[1]
        self.policy_network = PolicyNetwork(input_dim=self.num_features, lr=lr)

    def fit(self, num_epoches=1000, max_memory=60, balance=10000000,
            discount_factor=0, start_epsilon=.5, learning=True):
        self.policy_network.fit()


    def trade(self, model_path=None, balance=2000000):
        if model_path is None:
            return
        self.policy_network.load_model(model_path=model_path)
        self.fit(balance=balance, num_epoches=1, learning=False)
