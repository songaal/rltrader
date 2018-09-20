import os
import locale
import logging
import numpy as np
from keras.datasets import mnist

from no_agent.policy_network import PolicyNetwork

logger = logging.getLogger(__name__)
locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')


class PolicyLearner:

    def __init__(self, symbol, x_train, lr=0.01):
        self.symbol = symbol
        self.x_train = x_train  # 학습 데이터
        self.num_features = self.x_train.shape
        self.policy_network = PolicyNetwork(input_dim=self.num_features, lr=lr)

    def fit(self, x_train, y_train, x_test, y_test, num_epoches=1000, batch_size=10):

        self.policy_network.fit(x_train=x_train, y_train=y_train,
                                epochs=num_epoches, batch_size=batch_size,
                                x_test=x_test, y_test=y_test)

    def trade(self, x, model_path=None):
        if model_path is None:
            return
        self.policy_network.load_model(model_path=model_path)
        prob = self.policy_network.predict(x)
        # 예측 결과.. -1 ~ 1 사이의 값..?
        # 1 매수
        # -1 매도
        print(prob)
