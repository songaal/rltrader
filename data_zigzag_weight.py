import os

import pandas
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick2_ohlc
import matplotlib.ticker as ticker
import datetime as datetime
import numpy as np
import numpy as np
import pandas as pd
import zigzag

fpath = os.path.dirname(os.path.abspath(__file__))
fpath += '/data/ingest_data/'
load_file_name = 'up_down_binance_btc_usdt_4h.csv'
write_weight_file_name = 'weight_binance_btc_usdt_4h.csv'
chart_data = pd.read_csv(fpath + load_file_name, thousands=',', header=None)
chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'action']

offset = 1
weight_list = []
tmp_action = None
start_index = None
is_append = False

for i in range(1, len(chart_data['action'])):
    act = chart_data['action'][i]
    is_append = False
    if tmp_action is None:
        tmp_action = chart_data['action'][0]
        start_index = 0

    if tmp_action != act:
        print('range: {} ~ {}, count: {}, action: {} => {}'.format(start_index, i, (i - start_index), tmp_action, act))
        is_append = True
    elif i == len(chart_data['action']) - 1:
        print('last > range: {} ~ {}, count: {}, action: {} => {}'.format(start_index, i, (i - start_index), tmp_action, act))
        is_append = True

    if is_append:
        cnt = i - start_index
        for c in range(cnt):
            if c < offset:
                w = 0.0
            elif tmp_action == 'U':
                w = 1 / (cnt - 1)
            else:
                w = -(1 / (cnt - 1))
            weight_list.append(float(w))
        tmp_action = act
        start_index = i

weight_list = np.array(weight_list)
# 0번 데이터는 이전 데이터가 없으므로 삭제.
chart_data = chart_data.drop(0)
chart_data = chart_data.drop(columns='action')
chart_data['weight'] = np.array(weight_list)
chart_data.to_csv(fpath + write_weight_file_name, mode='w', header=False)
print('저장 완료.')


def ohlcv_polt(data):
    fig, ax = plt.subplots()
    date = np.array(data.date)
    open = np.array(data.open)
    high = np.array(data.high)
    low = np.array(data.low)
    close = np.array(data.close)
    volume = np.array(data.volume)
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


def plot_pivots(X, weight_list):
    X = np.array(X)
    plt.plot(np.arange(len(X))[weight_list == 0.0], np.array(X)[weight_list == 0.0], 'ro')
    # plt.plot(np.arange(len(X))[weight_list > 0.0], np.array(X)[weight_list > 0.0], color='b')


ohlcv_polt(chart_data)
plot_pivots(chart_data.high, weight_list)
plt.show()
