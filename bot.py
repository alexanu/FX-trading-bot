#import requests
#import json
#import time
#import sys
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#import mpl_finance as mpf
from scipy.stats import linregress

#User difined classes
from OandaEndpoints import Order, Position, Pricing, Instrument
from OandaCandleStick import CandleStick
#from predict_RNN import RNNPredictor
from RNNparam import RNNparam
from Fetter import Fetter
from Plotter import Plotter
from Evaluator import Evaluator
#from Predictor import Predictor
from Trader import Trader
from RoutineInspector import RoutineInspector

#User difined functions
from Notify import notify_from_line
from OandaEndpoints import from_byte_to_dict, from_response_to_dict

'''
Environment            Description
fxTrade(Live)          The live(real) environment
fxTrade Practice(Demo) The Demo (virtual) environment
'''

def accumulate_timeframe(response, candlestick, strategy):
	"""
	旧ver.の初期化関数
	candlestick内のcountをもとに指定された本数のローソク足をtickから生成する関数
	時間足*ローソク足の本数分の処理時間がかかるため、実用的ではないと判断。廃止

	Parameters
	----------
	recv: dict
		tickデータを含む受信データ
	candlestick: CandleStick
		ある時間足のローソク足データ
	"""

	#時間足ごとのbooleanリストを生成,すべての時間足のセットアップが完了したかの判定に使用
	flags = {k: False for k in candlestick.keys()}

	#現在所持しているポジションをすべて決済する
	strategy.clean()

	#Oandaサーバからtickデータを取得
	for line in response.iter_lines(1):
		print(line)
		if has_price(line):
			recv = from_byte_to_dict(line)

			### NEW PROCESS(This process separates time frames I deal with)
			for k, v in candlestick.items():
				if can_update(recv, v, mode='init') is True:
					v.update_ohlc()
					flags[k] = True

					#flagsの要素すべてがTrueになっているかを評価
					if list(flags.values).count(True) == len(flags):
						return None

				v.append_tickdata(recv)
		else:
			continue

def can_not_connect(res):
	if res.status_code != 200:
		return True
	else:
		return False

def can_update(recv, candlestick, mode=None):
	"""
	ローソク足が更新可能かを返す, 対象とする時間足を指定する必要はない
	for文などで予め対象とするローソク足を抽出する必要あり

	Parameters
	----------
	recv: dict
		tickデータを含む受信データ
	candlestick: CandleStick
		ある時間足のローソク足データ
	"""
	dummy_tick = pd.Series(float(recv["bids"][0]["price"]),index=[pd.to_datetime(recv["time"])])
	dummy_tickdata = candlestick.tickdata.append(dummy_tick)
	dummy_ohlc = dummy_tickdata.resample(candlestick.rate).ohlc()

	num_candle = candlestick.count if mode is 'init' else 0
	
	if((num_candle + 2) <= len(dummy_ohlc)):
		return True
	else:
		return False

def debug_print(src):
	print('<for debug>  {}'.format(src))

def driver(candlesticks, instrument, environment='demo'):
	"""
	ローソク足データの収集、解析、取引を取り扱う

	Parameters
	----------
	candlesticks: dict
		ローソク足データを任意の時間足分格納したdict
	environment: str
		取引を行う環境。バーチャル口座orリアル口座
	instrument: str
		取引を行う通貨	
	"""
	strategy_handler = Strategy(environment, instrument)
	#現在保有しているポジションをすべて決済
	strategy_handler.clean()

	#指定した通貨のtickをストリーミングで取得する
	pricing_handler = Pricing(environment)
	response = pricing_handler.connect_to_stream(instrument)
	if can_not_connect(response) == True:
		print('failed to connect with Oanda Streaming API')
		print(response.text)
		return

	#oanda_visualizer = Visualizer()
	for line in response.iter_lines(1):
		if has_price(line):
			recv = from_byte_to_dict(line)

			### NEW PROCESS(This process separates time frames I deal with)
			for v in candlesticks.values():
				if can_update(recv, v) is True:
					v.update_ohlc()
				v.append_tickdata(recv)
		else:
			continue

def has_price(msg):
	msg = from_byte_to_dict(msg)
	if msg["type"] == "PRICE":
		return True
	else:
		return False

def initialize(timeframes, instrument, environment='demo'):
	debug_print('Initialize start')
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

	#timeframsの情報をもとにCandleStickを生成,時間足をキーとして地所型に格納
	candlesticks = {t: CandleStick(t, c) for t, c in timeframes.items()}

	#APIを叩くhandler呼び出し
	instrument_handler = Instrument(environment)
	
	#任意のすべての時間足において、指定された本数のローソク足を取得
	for k, v in candlesticks.items():
		#各時間足ごとのローソク足のデータを取得
		resp = instrument_handler.fetch_candle(instrument, 'M', offset_table[k], v.count)
		debug_print(resp)

		#接続失敗した場合、Noneを返し終了
		if can_not_connect(resp) == True:
			print(resp.text)
			return None
		debug_print('Pricing handler get connection')

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
	debug_print('initialize end')
	return candlesticks

def all_element_in(signals):
	"""
	リストに格納されているbool値がすべてTrueかすべてFalseかを判断して返す。
	TrueとFalseが混ざっている場合はNoneを返す

	Parameters
	----------
	signals: list(boolean list)
		任意の時間足におけるエントリーをするorしないを格納したリスト
	"""
	#numpyのall()メソッドを使うと処理が楽になるかもしれない
	is_all_true = (list(signals.values()).count(True) == len(signals))
	is_all_false = (list(signals.values()).count(False) == len(signals))

	#is_all_trueのみがTrue、またはis_all_falseのみがTrueのときを判定
	if False is is_all_true and False is is_all_false:
		debug_print('All timeframe tendency do not correspond')
		return None
	elif True is is_all_true and False is is_all_false:
		return True
	elif False is is_all_true and True is is_all_false:
		return False
	else:
		debug_print('This section may have mistakes.')
		debug_print('Stop running and Do debug.')
		return None

def calc_trendline(candlesticks, price='high'):
	"""
	ローソク足をもとにトレンドラインを自動生成する

	Parameters
	----------
	candlesticks: dict
		ローソク足データを任意の時間足分格納したdict
	price: str
		トレンドラインの生成に参照される価格を選択する。選択できるのは高値('high')or安値('low')
	"""
	slopes = {}
	intercepts = {}
	ohlc = {}
	for k, v in candlesticks.items():
		ohlc[k] = v.ohlc.copy()
		ohlc[k] = (ohlc[k] - ohlc[k].min()) / (ohlc[k].max() - ohlc[k].min())
		ohlc[k]['time_id'] = np.array([i+1 for i in range(len(ohlc[k]))])
		while len(ohlc[k]) > 3:
			x = ohlc[k]['time_id']
			y = ohlc[k][price]
			slopes[k], intercepts[k], _, _, _ = linregress(x, y)

			if price == 'high':
				left_hand = ohlc[k][price]
				right_hand = slopes[k] * x + intercepts[k]
			elif price == 'low':
				left_hand = slopes[k] * x + intercepts[k]
				right_hand = ohlc[k][price]
			else:
				print('input illegal parameter in price. only high or low')

			ohlc[k] = ohlc[k].loc[left_hand > right_hand]
	return slopes, intercepts

def test_driver(candlesticks, instrument, environment='demo'):
	"""
	ローソク足データの収集、解析、取引を取り扱う

	Parameters
	----------
	candlesticks: dict
		ローソク足データを任意の時間足分格納したdict
	environment: str
		取引を行う環境。バーチャル口座(demo)orリアル口座(live)
	instrument: str
		取引を行う通貨	
	"""
	debug_print('test_driver begin')
	mode = 'test'

	#予測器クラスのインスタンスを生成
	#時間足ごとに生成
	#predictors = {
	#	k: Predictor(k) for k in candlesticks.keys()
	#}
	#debug_print('predictor was created')


	#注文用クラスのインスタンスを生成
	trader = Trader(instrument, environment, mode)
	debug_print('trader was created')

	#現在保有しているポジションをすべて決済
	trader.clean()
	debug_print('close all position to use Strategy.clean()')

	#足かせクラスのインスタンスを生成
	#時間足ごとに生成
	#fetters = {
	#	k: Fetter(k) for k in candlesticks.keys()
	#}
	#debug_print('fetter was created')

	#strategy_handler = Strategy(environment, instrument)
	#現在保有しているポジションをすべて決済
	#strategy_handler.clean()
	#debug_print('close all position to use Strategy.clean()')

	#指定した通貨のtickをストリーミングで取得する
	pricing_handler = Pricing(environment)
	response = pricing_handler.connect_to_stream(instrument)
	if can_not_connect(response) == True:
		print('failed to connect with Oanda Streaming API')
		print(response.text)
		return
	debug_print('pricing handler can connect with Pricing API')

	#定期的に走るルーチンを制御するためのクラス
	#注文と決済が通っているかheartbeat10回おきに監視するために使用
	routiner = RoutineInspector(freq=10)

	#評価用クラスのインスタンスを生成
	timeframes = list(candlesticks.keys())
	evaluator = Evaluator(timeframes, instrument, environment)

	#oanda_visualizer = Visualizer()
	for line in response.iter_lines(1):
		if has_price(line):
			recv = from_byte_to_dict(line)
			#debug_print(recv)

			### NEW PROCESS(This process separates time frames I deal with)
			for k, v in candlesticks.items():
				#print(v.normalize_by('close', raw=False))
				print(v.normalize_by('close').values)
				if can_update(recv, v) is True:
					v.update_ohlc_()
					print(k)
					print(v.ohlc)
					print(len(v.ohlc))

					#エントリー
					entry(k, candlesticks, trader, evaluator)
					#決済（クローズ）
					settle(k, candlesticks, trader, evaluator)

					#時間足が更新されたときにも注文が反映されたかを確認する
					#注文が反映されたか
					if trader.test_is_reflected_order() is True:
						#WAIT_ORDER -> POSITION
						trader.switch_state()

					#時間足が更新されたときにも決済が反映されたかを確認する
					#決済が反映されたか
					if trader.test_is_reflected_position() is True:
						#WAIT_POSITION -> ORDER
						trader.switch_state()

				v.append_tickdata(recv)
		else:
			pass

		#一定時間ごとに注文or決済が反映されたかを確認する
		if routiner.is_inspect() is True:
			print(f'heart beat(span): {routiner.count}')
			#注文が反映されたか
			if trader.test_is_reflected_order() is True:
				#WAIT_ORDER -> POSITION
				trader.switch_state()

			#決済が反映されたか
			if trader.test_is_reflected_position() is True:
				#WAIT_POSITION -> ORDER
				trader.switch_state()
		routiner.update()

def entry(key, candlesticks, trader, evaluator):	
	"""
	----------------
	4時間足の最後の2本がしましまのときにエントリー
	"""

	### Algorythm begin ###
	if key == '15min' and trader.state == 'ORDER':
		#Ichimatsu Strategy
		slopes = {}
		intercepts = {}
		num_sets = {}
		ohlc = candlesticks['4H'].ohlc.copy()
		ohlc = (ohlc - ohlc.values.min()) / (ohlc.values.max() - ohlc.values.min())

		#ローソク足を2本1セットとして、numに対となるローソク足の本数を指定
		num_sets['4H'] = 2
		tail = ohlc[-num_sets['4H']:]

		close_open = ['close' if i%2 == 0 else 'open' for i in range(num_sets['4H'])]
		x = np.array([i + 1 for i in range(num_sets['4H'])])
		y = [tail[co].values[i] for i, co in enumerate(close_open)]

		#meno stderrを使うかもしれない
		slopes['4H'], intercepts['4H'], _, _, _ = linregress(x, y)

		bottom = [+1 if i%2 == 0 else -1 for i in range(num_sets['4H'])]
		top    = [+1 if i%2 != 0 else -1 for i in range(num_sets['4H'])]
		signs = list(np.sign(tail['open'].values - tail['close'].values))

		print(f'Top: {top==signs}')
		print(f'Bottom: {bottom==signs}')

		kind = None
		is_rise = None
		#and np.abs(slopes['5min']) < 0.01

		#底値
		if bottom == signs:
			#BUY
			is_rise = True
			kind = 'BUY'
			print('Bottom')
		#高値
		elif top == signs:
			#SELL
			is_rise = False
			kind = 'SELL'
			print('Top')
		else:
			print('Unsteable')


		if kind is not None:
			print(kind)
			is_order_created = trader.test_create_order(is_rise)
			#評価関数に注文情報をセット
			evaluator.set_order(kind, True)

			evaluator.begin_plotter()
			evaluator.add_candlestick(candlesticks)
			evaluator.add_tail_oc_slope(candlesticks, slopes, intercepts, num_sets)
			evaluator.add_ichimatsu(candlesticks, num_sets)
			evaluator.end_plotter('signal.png', True)

			if True is is_order_created:
				#ORDER状態からORDERWAITINGに状態遷移
				trader.switch_state()
			else:
				print('order was not created')

		else:
			evaluator.begin_plotter()
			evaluator.add_candlestick(candlesticks)
			evaluator.add_tail_oc_slope(candlesticks, slopes, intercepts, num_sets)
			evaluator.add_ichimatsu(candlesticks, num_sets)
			evaluator.end_plotter('signal.png', False)
			notify_from_line('progress', image='signal.png')

	### Algorythm end ###


def settle(key, candlesticks, trader, evaluator):
	"""
	-------------------------
	#決済を行うかどうかを判断
	#(now)15分足が2本更新された際に決済を行う
	"""
	if key == '15min' and trader.state == 'POSITION':
		#ポジションを決済可能か
		if trader.can_close_position() is True:
			#決済の注文を発行する
			is_position_closed = trader.test_close_position()

			x = {}
			correct = {}
			for k, v in candlesticks.items():
				x[k] = v.normalize_by('close').values
				#or x[k] = v.normalize_by('close' raw=True)

				#要修正
				threshold = 2
				correct[k] = np.mean(x[k][-threshold:])

			evaluator.set_close()

			#LINE notification
			evaluator.output_close()
			evaluator.log_score()

			#決済の注文が発行されたか
			if is_position_closed is True:
				#ORDER -> WAIT_ORDER
				trader.switch_state()
			else:
				pass
		else:
			debug_print('Position can not be closed in this update')
		#決済するかを判断するアルゴリズムを更新
		trader.update_whether_closing_position()

def main():
	timeframes = {
		'15min': RNNparam('15min').tau,
		'4H': RNNparam('4H').tau
	}

	instrument = 'GBP_JPY'
	environment = 'demo'

	#初期化	
	candlesticks = initialize(timeframes, instrument, environment)
	#APIとの接続が失敗した場合Noneが返り終了
	if candlesticks is None:
		print('failed to connect with Oanda Instrument API')
		return

	#driver(candlesticks, instrument, environment)
	test_driver(candlesticks, instrument, environment)

if __name__ == "__main__":
	main()
