import numpy as np
import math

class Agent:
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 2  # 주식 보유 비율, 포트폴리오 가치 비율

    # 매매 수수료
    TRADING_CHARGE = 0.001  # 거래 수수료
    SLIPPAGE = 0.0004  # 슬리피지

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩
    ACTIONS = [ACTION_BUY, ACTION_SELL]  # 인공 신경망에서 확률을 구할 행동들
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(
        self, environment, min_trading_unit=1, max_trading_unit=2, 
        delayed_reward_threshold=.05):
        # Environment 객체
        self.environment = environment  # 현재 주식 가격을 가져오기 위해 환경 참조

        # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위
        self.delayed_reward_threshold = delayed_reward_threshold  # 지연보상 임계치

        # Agent 클래스의 속성
        self.initial_balance = 0  # 초기 자본금
        self.balance = 0  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        self.portfolio_value = 0  # balance + num_stocks * {현재 주식 가격}
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks / int(
            self.portfolio_value / self.environment.get_price())
        self.ratio_portfolio_value = self.portfolio_value / self.base_portfolio_value
        return (
            self.ratio_hold,
            self.ratio_portfolio_value
        )

    def decide_action(self, policy_network, sample, epsilon):
        confidence = 0.
        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)  # 무작위로 행동 결정
        else:
            exploration = False
            probs = policy_network.predict(sample)  # 각 행동에 대한 확률
            action = np.argmax(probs)
            confidence = probs[action]
        return action, confidence, exploration

    def validate_action(self, action):
        validity = True
        if action == Agent.ACTION_BUY:
            # self.environment.get_price() 종가.
            # (1 + self.TRADING_CHARGE) 1개 수수료
            # self.min_trading_unit 최소 구매 단위
            if self.balance < (self.environment.get_price() * (1 + self.TRADING_CHARGE) * self.min_trading_unit) * (1 + self.SLIPPAGE):
                validity = False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인 
            if self.num_stocks <= 0:
                validity = False
        return validity

    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        added_traiding = max(min(
            int(confidence * (self.max_trading_unit - self.min_trading_unit)),
            self.max_trading_unit-self.min_trading_unit
        ), 0)
        return self.min_trading_unit + added_traiding

    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            commission = curr_price * self.TRADING_CHARGE
            slipage = curr_price * self.SLIPPAGE
            buy_price = curr_price + commission + slipage
            trading_unit = math.floor((self.balance / buy_price) * 100000000) / 100000000
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = buy_price * trading_unit
            # print('BOT: 보유 자산: ', self.balance, '종가:', curr_price, ', 구매 수량: ', trading_unit, '총구매 가격:', invest_amount, '보유코인 수량:', self.num_stocks)
            self.balance -= invest_amount  # 보유 현금을 갱신
            self.num_stocks += trading_unit  # 보유 코인 수를 갱신
            self.num_buy += 1  # 매수 횟수 증가
        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.num_stocks  # 전체 수량
            sell_price = self.num_stocks * curr_price  # 전체수량 판매가격
            commission = sell_price * self.TRADING_CHARGE  # 수수료 제외
            slipage = sell_price * self.SLIPPAGE  # 슬리피지 제외
            # 매도
            invest_amount = sell_price - commission - slipage
            # print('SLD: 보유 자산: ', self.balance, '종가:', curr_price, ', 판매 수량: ', trading_unit, '총판매 가격:', invest_amount, '보유코인 수량: ', self.num_stocks)
            self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
            self.balance += invest_amount  # 보유 현금을 갱신
            self.num_sell += 1  # 매도 횟수 증가

        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        profitloss = (
            (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)

        # 즉시 보상 판단
        self.immediate_reward = 1 if profitloss >= 0 else -1

        # 지연 보상 판단
        if profitloss > self.delayed_reward_threshold:
            delayed_reward = 1
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        elif profitloss < -self.delayed_reward_threshold:
            delayed_reward = -1
            # 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        else:
            delayed_reward = 0
        return self.immediate_reward, delayed_reward
