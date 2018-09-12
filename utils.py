# import csv
# from datetime import datetime, timedelta
# import json
# import logging
#
# import os
# import tempfile
# import timeit
#
# import re
# import requests
#
# logging.basicConfig(format='[%(asctime)s %(levelname)s] (%(filename)s:%(lineno)d) %(message)s',
#                     level=os.environ.get('LOGLEVEL', 'DEBUG'))
#
# # Name the logger after the package.
# logger = logging.getLogger(__package__)
#
#
# def split_interval(time_interval):
#     """
#     2h, 1d, 30m 과 같이 입력된 인터벌을 분리한다.
#     :param time_interval:
#     :return:
#     """
#     unit = None
#     number = int(re.findall('\d+', time_interval)[0])
#     maybeAlpha = time_interval[-1]
#     if maybeAlpha.isalpha():
#         unit = maybeAlpha.lower()
#     return number, unit
#
#
# def ingest_filename(symbol, period, history):
#     return '{}_{}_{}.csv'.format(symbol.replace('/', '_').lower(), period, history)
#
#
# def ingest_filepath(root_dir, exchange, symbol, start_date, end_date, period, history):
#     filename = ingest_filename(symbol, period, history)
#     base_dir = '{}/{}/{}-{}'.format(root_dir,
#                                     exchange,
#                                     start_date.strftime('%Y%m%d%H%M%Z'),
#                                     end_date.strftime('%Y%m%d%H%M%Z')
#                                     )
#     try:
#         os.makedirs(base_dir, exist_ok=True)
#     except OSError as e:
#         raise e
#
#     return base_dir, os.path.join(base_dir, filename)
#
#
# def ingest_data(exchange, symbol, start, end, interval):
#     api_gateway_endpoint = 'https://9u3jawxuod.execute-api.ap-northeast-2.amazonaws.com/v1_1'
#
#     # 값, 단위 분리
#     interval_num, interval_unit = split_interval(interval)
#     interval = interval.lower()
#     interval_unit = interval_unit.lower()
#
#     if interval_unit in ['w', 'd', 'h']:
#         # 주, 일, 시 단위
#         resolution = interval_unit if interval_num == '1' else interval
#     elif interval_unit in ['m']:
#         # 분 단위
#         resolution = interval_num
#
#     url = '{}/history'.format(api_gateway_endpoint)
#     params = {'resolution': resolution, 'from': start, 'to': end,
#               'symbol': symbol.upper(), 'exchange': exchange}
#     logger.debug('Get candle: %s > %s', url, params)
#     response = requests.request('GET', url, params=params)
#     candles = json.loads(response.text)
#
#     f = open('./data/chart_date/', 'w', encoding='utf-8', newline='')
#     wr = csv.writer(f)
#
#     if len(candles['t']) == 0:
#         raise ValueError('[FAIL] candle data row 0')
#
#     wr.writerow(['index', 'ts', 'open', 'high', 'low', 'close', 'volume'])
#     for index in range(len(candles['t'])):
#         if candles['o'][index] and candles['h'][index] and candles['l'][index] and candles['c'][index] and \
#                 candles['v'][index]:
#             time = datetime.fromtimestamp(int(candles['t'][index]), tz=tz).strftime('%Y-%m-%d')
#             wr.writerow([
#                 time,
#                 '{:d}'.format(candles['t'][index]),
#                 '{:.8f}'.format(candles['o'][index]),
#                 '{:.8f}'.format(candles['h'][index]),
#                 '{:.8f}'.format(candles['l'][index]),
#                 '{:.8f}'.format(candles['c'][index]),
#                 '{:.2f}'.format(candles['v'][index]),
#             ])
#     timer_end = timeit.default_timer()
#     logger.debug('# {} Downloaded CandleFile. elapsed: {}'.format(symbol, str(timer_end - timer_start)))
#     return base_dir