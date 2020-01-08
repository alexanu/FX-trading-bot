import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd

#User difined classes
from Manager import Manager
from CommonParam import CommonParam
#from OandaEndpoints import Order, Position, Pricing, Instrument
from OandaEndpoints import Instrument
from OandaCandleStick import CandleStick
#from predict_RNN import RNNPredictor
#from RNNparam import RNNparam
#from Fetter import Fetter
#from Plotter import Plotter
#from Predictor import Predictor
#from Trader import Trader
#from Evaluator import Evaluator
#from PlotHelper import PlotHelper
#from RoutineInspector import RoutineInspector

#User difined functions
from Notify import notify_from_line
from OandaEndpoints import from_byte_to_dict, from_response_to_dict
from Analysis import is_rise_with_zebratail, is_rise_with_trendline

'''
Environment            Description
fxTrade(Live)          The live(real) environment
fxTrade Practice(Demo) The Demo (virtual) environment
'''

#def can_update(recv, candlestick, mode=None):
#	"""
#	ローソク足が更新可能かを返す, 対象とする時間足を指定する必要はない
#	for文などで予め対象とするローソク足を抽出する必要あり
#
#	Parameters
#	----------
#	recv: dict
#		tickデータを含む受信データ
#	candlestick: CandleStick
#		ある時間足のローソク足データ
#	"""
#	dummy_tick = pd.Series(float(recv["bids"][0]["price"]),index=[pd.to_datetime(recv["time"])])
#	dummy_tickdata = candlestick.tickdata.append(dummy_tick)
#	dummy_ohlc = dummy_tickdata.resample(candlestick.rate).ohlc()
#
#	num_candle = candlestick.count if mode is 'init' else 0
#	
#	if((num_candle + 2) <= len(dummy_ohlc)):
#		return True
#	else:
#		return False
#
#def has_price(msg):
#	if msg:
#		msg = from_byte_to_dict(msg)
#		if msg["type"] == "PRICE":
#			return True
#		else:
#			return False
#	else:
#		return False
#
#def has_heartbeat(msg):
#	if msg:
#		msg = from_byte_to_dict(msg)
#		if msg["type"] == "HEARTBEAT":
#			return True
#		else:
#			return False
#	else:
#		return False

def initialize(timeframes, instrument, environment='demo'):
	print('Initialize start')
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

	#timeframesの情報をもとにCandleStickを生成,時間足をキーとして地所型に格納
	candlesticks = {t: CandleStick(t, c) for t, c in timeframes.items()}

	#APIを叩くhandler呼び出し
	instrument_handler = Instrument(environment)
	
	#任意のすべての時間足において、指定された本数のローソク足を取得
	for k, v in candlesticks.items():
		#各時間足ごとのローソク足のデータを取得
		resp = instrument_handler.fetch_candle(instrument, 'M', offset_table[k], v.count)
		print(resp)

		#接続失敗した場合、Noneを返し終了
		try:
			resp.raise_for_status()
		except ConnectionError as e:
			print(f'{e}')
			return None
		print('Pricing handler get connection')

		#取得したローソク足データをdict型へ変換
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
		print(v.ohlc)
		print(len(v.ohlc.index))
	print('initialize end')
	return candlesticks

def main():
	#timeframes = {
	#	'5min': RNNparam('5min').tau,
	#	'15min': RNNparam('15min').tau
	#}

	instrument = 'GBP_JPY'
	environment = 'demo'
	mode = 'test'
	param = CommonParam()
	manager = Manager(param, instrument, environment, mode)

	#初期化	
	candlesticks = initialize(param.timeframes, instrument, environment)
	#APIとの接続が失敗した場合Noneが返り終了
	if candlesticks is None:
		print('failed to connect with Oanda Instrument API')
		return

	manager.driver(candlesticks)

	print('driver is over')
if __name__ == "__main__":
	main()
