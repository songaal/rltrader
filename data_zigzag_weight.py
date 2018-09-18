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
chart_data = pd.read_csv(fpath + load_file_name, thousands=',', header=None)
chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'action']
# chart_data = chart_data[(chart_data['date'] >= '2018-09-15') & (chart_data['date'] <= '2018-09-17')]


weight_list = []
tmp_action = None
start_index = None
for i in range(len(chart_data['action'])):
    print(chart_data['action'][i])
    act = chart_data['action'][i]

    if tmp_action is None:
        tmp_action = act
        start_index = 0
        continue

    if act == 'U':

        start_index



# print(chart_data)
