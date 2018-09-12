# rltrader 사용방법

## 학습방법
- 명령어 python main_coin.py
- 모델파일은 models/{symbol}/model_{model_ver}.h5 생성

## 실행방법
- main_notraining_coin.py 파일에서 symbol, model_ver를 수정.
- 명령어 python main_notraining_coin.py

## 심볼 변경방볍
- 데이터 다운로드는 influxdb에서 쿼리 검색후 csv 저장파일을 rltrader/data/chart_data/{심볼명}.csv에 위치해야합니다.
- main_con.py 파일에서 symbol 변수에 파일명과 동일하게 입력

## 학습데이터 범위변경방법
- main_con.py 파일에서 t_start, t_end 기간을 설정.

## 수수료 변경
- agent_coin.py 파일에서 TRADING_CHARGE 값 수정.


