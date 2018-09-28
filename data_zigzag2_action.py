import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import zigzag
from matplotlib.finance import candlestick2_ohlc

fpath = os.path.dirname(os.path.abspath(__file__))
fpath += '/data/ingest_data/'
load_file_name = 'binance_btc_usdt_4h.csv'
write_up_down_file_name = 'action_binance_btc_usdt_4h.csv'
chart_data = pd.read_csv(fpath + load_file_name, thousands=',', header=None)
chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

chart_data['date'] = pd.to_datetime(chart_data['date'])
chart_data = chart_data[(chart_data['date'] >= '2017-11-01') & (chart_data['date'] <= '2018-02-01')]

high_low = []
trend = 0

open_a = chart_data.open.values
close_a = chart_data.close.values
low_a = chart_data.low.values
high_a = chart_data.high.values

ohlcv4_a = (open_a + close_a + low_a + high_a) / 4

for i in range(len(chart_data.date)):
    open = open_a[i]
    close = close_a[i]
    low = low_a[i]
    high = high_a[i]
    if i == 0:
        high_low.append(high if open < close else low)
        continue
    high_low.append(max(ohlcv4_a[i], ohlcv4_a[i - 1]))

X = np.array(high_low)
pivots = zigzag.peak_valley_pivots(X, 0.02, -0.01)
"""
위 변곡점: 1
아래 변곡점: -1
나머지: 0

swsong
위 꼭지점은 -1 
아래 꼭지점은 1
번갈아 나오면 0점.


"""

hold_count = 1
left_hold_range = 0.15
right_hold_range = 0.01

# action = 'B', 'S', 'H'
actions = []

last_action = None
last_action_price = None
last_action_index = 0
highest_price = 0
lowest_price = 0
prev_pivot_index = 0
tmp_pivot = None
current_hold_count = hold_count

def get_next_pivot_index(index, hold_count):
    next_index = None
    for i in range(index + hold_count, len(pivots)):
        if pivots[i] != 0:
            next_index = i
            break
    return next_index


for index in range(len(pivots)):
    price = close_a[index]
    pivot = pivots[index]
    act = None

    if last_action is None:
        # 처음엔 상태가 없으므로 매수
        act = 'B'
        tmp_pivot = 1
    elif pivot != 0 or current_hold_count > 0:
        # 홀드봉이 있을 경우.
        # pivot 0 아닌 경우
        act = 'H'
        current_hold_count -= 1
        if pivot != 0:
            tmp_pivot = pivot
            prev_pivot_index = index
            if current_hold_count <= 0:
                current_hold_count = hold_count
    else:
        next_pivot_index = get_next_pivot_index(index, 1)
        if next_pivot_index is None:
            act = 'H'
        else:
            print('--------------------------------------------')
            print('date: {}'.format(chart_data['date'].values[index]))

            total_count = next_pivot_index - prev_pivot_index
            current_count = index - prev_pivot_index

            act = 'H'
            if tmp_pivot == -1:
                is_left_hold_action = (total_count - current_count) / total_count < left_hold_range
                is_right_hold_action = abs(lowest_price - price) / price < right_hold_range
                print('매수')
                print('left: {}, right: {}'.format(is_left_hold_action, is_right_hold_action))
                if is_left_hold_action or is_right_hold_action:
                    act = 'H'
                else:
                    act = 'B'
            if tmp_pivot == 1:
                is_left_hold_action = (total_count - current_count) / total_count < left_hold_range
                is_right_hold_action = (highest_price - price) / price < right_hold_range
                print('매도')
                print('left: {}, right: {}'.format(is_left_hold_action, is_right_hold_action))
                if is_left_hold_action or is_right_hold_action:
                    act = 'H'
                else:
                    act = 'S'

            print('act: {}, trends: {}'.format(act, tmp_pivot))
            print('price: {}, lowest: {}, lowest: {}'.format(price, lowest_price, lowest_price))
            print('--------------------------------------------')

    if highest_price < price:
        highest_price = price
    if lowest_price > price:
        lowest_price = price

    if act != 'H':
        last_action = act
        last_action_price = price
        last_action_index = index
        highest_price = price
        lowest_price = price

    actions.append(act)

actions = np.array(actions)

fake_data = zip(range(len(actions)), chart_data.date, actions)
act_data = pd.DataFrame([data for num, *data in fake_data], columns=['date', 'action'])
act_one_hot = pd.get_dummies(act_data)
chart_data = pd.merge(chart_data, act_one_hot, on='date')
# B, H, S
chart_data.to_csv(fpath + write_up_down_file_name, mode='w', index=False, header=False)
print(chart_data.head(10))
print('저장 완료. {}'.format(fpath + write_up_down_file_name))

def ohlcv_plot(data):
    fig, ax = plt.subplots()
    date = data.date.values
    open = data.open.values
    high = data.high.values
    low = data.low.values
    close = data.close.values
    volume = data.volume.values
    candlestick2_ohlc(ax, open, high, low, close, width=0.6)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
    def mydate(x, pos):
        try:
            return pd.to_datetime(date[int(x)]).strftime('%Y.%m.%d %H:%M')
        except IndexError:
            return ''
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(mydate))
    fig.autofmt_xdate()
    fig.tight_layout()


def plot_pivots(X, pivots):
    plt.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
    # plt.scatter(np.arange(len(X))[pivots == 1], X[pivots == 1], color='g')
    # plt.scatter(np.arange(len(X))[pivots == -1], X[pivots == -1], color='r')

def plot_actions(X, actions):
    plt.scatter(np.arange(len(X))[actions == 'B'], X[actions == 'B'], color='g')
    plt.scatter(np.arange(len(X))[actions == 'S'], X[actions == 'S'], color='r')
    # plt.scatter(np.arange(len(X))[actions == 'H'], X[actions == 'H'], color='b')


ohlcv_plot(chart_data)
plot_pivots(X, pivots)
plot_actions(X, actions)
plt.show()

