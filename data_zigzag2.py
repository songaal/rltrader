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
write_up_down_file_name = 'up_down_binance_btc_usdt_4h.csv'
chart_data = pd.read_csv(fpath + load_file_name, thousands=',', header=None)
chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

# chart_data['date'] = pd.to_datetime(chart_data['date'])
# chart_data = chart_data[(chart_data['date'] >= '2018-09-01') & (chart_data['date'] <= '2018-09-17')]

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
    # ohlcv4 = ohlcv4_a[-1] #open + high + low + close / 4
    if i == 0:
        high_low.append(high if open < close else low)
        continue

    # p_open = open_a[i - 1]
    # p_close = close_a[i - 1]
    # p_low = low_a[i - 1]
    # p_high = high_a[i - 1]
    # p_ohlcv4 = ohlcv4[i - 1]

    # if p_ohlcv4 < ohlcv4:
    high_low.append(max(ohlcv4_a[i], ohlcv4_a[i - 1]))

    # if ohlcv4_a[i - 1] < ohlcv4_a[i]:
    #     high_low.append(high)
    # else:
    #     high_low.append(low)

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
actions = []
action = None
for pivot in pivots:
    if pivot == 1:
        action = 'D'
    elif pivot == -1:
        action = 'U'

    actions.append(action)

chart_data['actions'] = np.array(actions)
chart_data.to_csv(fpath + write_up_down_file_name, mode='w', index=False, header=False)
print('저장 완료.')

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
    # plt.xlim(0, len(X))
    # plt.ylim(X.min()*0.99, X.max()*1.01)
    # # plt.plot(np.arange(len(X)), X, 'k:', alpha=0.5)
    plt.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
    plt.scatter(np.arange(len(X))[pivots == 1], X[pivots == 1], color='g')
    plt.scatter(np.arange(len(X))[pivots == -1], X[pivots == -1], color='r')


ohlcv_plot(chart_data)
plot_pivots(X, pivots)
plt.show()

