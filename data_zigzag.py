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
load_file_name = 'binance_btc_usdt_4h.csv'
write_up_down_file_name = 'up_down_binance_btc_usdt_4h.csv'
chart_data = pd.read_csv(fpath + load_file_name, thousands=',', header=None)
chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

chart_data['date'] = pd.to_datetime(chart_data['date'])
# chart_data = chart_data[(chart_data['date'] >= '2018-09-01') & (chart_data['date'] <= '2018-09-17')]

high_low = []
trend = 0
for i in range(len(chart_data.date)):
    open = np.array(chart_data.open)[i]
    close = np.array(chart_data.close)[i]
    low = np.array(chart_data.low)[i]
    high = np.array(chart_data.high)[i]

    # high_low.append(close)
    if close >= open:
        high_low.append(high)
    else:
        high_low.append(low)

X = np.array(high_low)
pivots = zigzag.peak_valley_pivots(X, 0.01, -0.01)
"""
위 변곡점: 1
아래 변곡점: -1
나머지: 0
"""
actions = []
action = None
for pivot in pivots:
    if pivot == 1:
        action = 'D'
        pass
    elif pivot == -1:
        action = 'U'
        pass
    actions.append(action)

chart_data['actions'] = np.array(actions)
chart_data.to_csv(fpath + write_up_down_file_name, mode='w', index=False, header=False)
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


def plot_pivots(X, pivots):
    # plt.xlim(0, len(X))
    # plt.ylim(X.min()*0.99, X.max()*1.01)
    # # plt.plot(np.arange(len(X)), X, 'k:', alpha=0.5)
    plt.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
    plt.scatter(np.arange(len(X))[pivots == 1], X[pivots == 1], color='g')
    plt.scatter(np.arange(len(X))[pivots == -1], X[pivots == -1], color='r')


ohlcv_polt(chart_data)
plot_pivots(X, pivots)
plt.show()

