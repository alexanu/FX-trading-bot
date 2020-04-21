import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd

#User difined classes
from Manager import Manager
from CommonParam import CommonParam
from OandaEndpoints import Instrument
from OandaCandleStick import CandleStick
from Notify import notify_from_line
from OandaEndpoints import from_byte_to_dict, from_response_to_dict

def initialize(timeframes: dict, instrument: str, environment: str='demo') -> dict:
	offset_table = {
		'5S': 'S5',
		'10S': 'S10',
		'15S': 'S15',
		'30S': 'S30',
		'1min': 'M1',
		'2min': 'M2',
		'4min': 'M4',
		'5min': 'M5',
		'10min': 'M10',
		'15min': 'M15',
		'30min': 'M30',
		'1T': 'M1',
		'2T': 'M2',
		'4T': 'M4',
		'5T': 'M5',
		'10T': 'M10',
		'15T': 'M15',
		'30T': 'M30',
		'1H': 'H1',
		'2H': 'H2',
		'3H': 'H3',
		'4H': 'H4',
		'6H': 'H6',
		'8H': 'H8',
		'12H': 'H12',
		'1D': 'D',
		'1W': 'W',
		'1M': 'M'
	}

	#解析に用いるローソク足を定義
	candlesticks = {t: CandleStick(t, c) for t, c in timeframes.items()}
	#トレード環境（本番 or 仮想）
	instrument_handler = Instrument(environment)
	
	#ローソク足データの初期値を設定
	for k, v in candlesticks.items():
		resp = instrument_handler.fetch_candle(instrument, 'M', offset_table[k], v.count)
		print(resp)

		#接続可能かを検証
		resp.raise_for_status()
#		try:
#			resp.raise_for_status()
#		except ConnectionError as e:
#			print(f'{e}')
#			return None

		print('Instrument_handler get connection in Initialize from bot.py')

		#取得したローソク足を辞書型へ変換
		fetched_data = from_response_to_dict(resp)

		time_index = []
		default_ohlc = {
			'open': [],
			'high': [],
			'low': [],
			'close': []
		}

		for i in range(v.count):
			#responseから必要なデータを抽出し、順番にリストに格納する
			time_index.append(pd.to_datetime(fetched_data['candles'][i]['time']))
			default_ohlc['open'].append(float(fetched_data['candles'][i]['mid']['o']))
			default_ohlc['high'].append(float(fetched_data['candles'][i]['mid']['h']))
			default_ohlc['low'].append(float(fetched_data['candles'][i]['mid']['l']))
			default_ohlc['close'].append(float(fetched_data['candles'][i]['mid']['c']))

		#抽出したローソク足データを自作クラスのOHLCに代入
		ohlc = pd.DataFrame(default_ohlc, index=time_index)
		located_ohlc = ohlc.loc[:,['open', 'high', 'low', 'close']]
		v.ohlc = located_ohlc
		print(f'{k}: {len(v.ohlc.index)}')

	return candlesticks

def main():
	instrument = 'GBP_JPY'
	environment = 'demo'
	mode = 'test'
	param = CommonParam()
	manager = Manager(param, instrument, environment, mode)

	#取り扱うローソク足の初期値を設定
	try:
		candlesticks = initialize(param.timeframes, instrument, environment)
	except ConnectionError as e:
		print(f'{e}')
		print('failed to connect with Oanda Instrument API')
		return

	#ローソク足の初期化が心配しているかを検証
	#if candlesticks is None:
	#	print('failed to connect with Oanda Instrument API')
	#	return

	manager.driver(candlesticks)

if __name__ == "__main__":
	main()
