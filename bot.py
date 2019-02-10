import requests
import json
#import time
import sys
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
from Predictor import Predictor
from Trader import Trader
from RoutineInspector import RoutineInspector

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

def from_byte_to_dict(byte_line):
	"""
	byte型の文字列をdict型に変換する

	Parameters
	----------
	byte_line: byte
		byte型の文字列
	"""

	try:
		return json.loads(byte_line.decode("UTF-8"))
	except Exception as e:
		print("Caught exception when converting message into json : {}" .format(str(e)))
		return None

def from_response_to_dict(response):
	try:
		return json.loads(response.content.decode('UTF-8'))
	except Exception as e:
		print("Caught exception when converting message into json : {}" .format(str(e)))
		return None


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

def notify_from_line(message, image=None):
	url = 'https://notify-api.line.me/api/notify'
	with open(sys.argv[1]) as f:
		auth_tokens = json.load(f)
		token = auth_tokens['line_token']
		print(f'<bot> Read token from {sys.argv[1]}')

	headers = {
		'Authorization' : 'Bearer {}'.format(token)
	}

	payload = {
		'message' :  message
	}
	if image is not None:
		try:
			files = {
				'imageFile': open(image, "rb")
			}
			response = requests.post(url ,headers=headers ,data=payload, files=files)
			return response
		except:
			pass

	else:
		try:
			response = requests.post(url ,headers=headers ,data=payload)
			return response
		except:
			pass

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
	predictors = {
		k: Predictor(k) for k in candlesticks.keys()
	}
	debug_print('predictor was created')


	#注文用クラスのインスタンスを生成
	trader = Trader(instrument, environment, mode)
	debug_print('trader was created')

	#現在保有しているポジションをすべて決済
	trader.clean()
	debug_print('close all position to use Strategy.clean()')

	#足かせクラスのインスタンスを生成
	#時間足ごとに生成
	fetters = {
		k: Fetter(k) for k in candlesticks.keys()
	}
	debug_print('fetter was created')

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
				if can_update(recv, v) is True:
					v.update_ohlc_()
					print(k)
					print(v.ohlc)
					print(len(v.ohlc))
					print('###### Antlia Apricot Trading Strategy ######')

					"""
					Antria(アンチラ/ポンプ) Algorythm
					----------------
					5分足の最後の4本がしましまのときにエントリー

					"""
					entry_antlia(k, candlesticks, trader, evaluator)

					"""
					15分足のアルゴリズム
					-------------------
					15分足の更新時にstateが'ORDER'であるとき
					4時間足がトレンド（上昇or下降）かどうかを判定する
					T)4時間足が上昇トレンドだった場合
						T)15分足において,足かせを通るかをどうかを判定する
							T)買い注文
							F)注文を行わず処理終了
						F)15分足において,足かせを通るかをどうかを判定する
							T)売り:注文
							F)注文を行わず処理終了
					F)4時間足がレンジだった場合
						注文を行わず処理終了

					"""
					#entry_andromeda(key, candlesticks, trader, evaluator, predictors, fetters)


					"""
					5分足のアルゴリズム(debug)
					-------------------

					"""
					#entry_apus(key, candlesticks, trader, evaluator, predictors, fetters)


					"""
					#複数の時間足を考慮する際のアルゴリズム(の雛形)
					"""
					#entry_aquarius(key, candlesticks, trader, evaluator, predictors, fetters)


					"""
					Apricot(アンズ) Algorythm
					-------------------------
					#決済を行うかどうかを判断
					#(now)1分足が2本更新された際に決済を行う
					"""
					settle_apricot(k, candlesticks, trader, evaluator)
	
					"""
					Strawberry(イチゴ) Algorythm
					----------------------------
					#決済を行うかどうかを判断
					#5分足が2本更新された際に決済を行う
					"""
					#settle_strawberry(k, trader, evaluator)

					"""
					persimmon(カキ) Algorythm
					-------------------------
					#決済を行うかどうかを判断
					#(now)15分足が2本更新された際に決済を行う
					"""
					#settle_persimmon(k, trader, evaluator)

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


def entry_andromeda(key, candlesticks, trader, evaluator, predictors, fetters):
	"""
	15分足のアルゴリズム
	-------------------
	15分足の更新時にstateが'ORDER'であるとき
	4時間足がトレンド（上昇or下降）かどうかを判定する
	T)4時間足が上昇トレンドだった場合
		T)15分足において,足かせを通るかをどうかを判定する
			T)買い注文
			F)注文を行わず処理終了
		F)15分足において,足かせを通るかをどうかを判定する
			T)売り:注文
			F)注文を行わず処理終了
	F)4時間足がレンジだった場合
		注文を行わず処理終了

	"""
	### Algorythm begin ###
	if key == '15min' and trader.state == 'ORDER':

		x = {}
		x_rnn = {}
		predicted = {}
		
		#特定の時間足において予測を行う場合
		for k, v in candlesticks.items():
			#OHLCの中から終値をベクトルとして取り出す
			x[k] = predictors[k].from_ohlc_to_vector(v.ohlc, 'close')
			#取り出した終値ベクトルを正規化する
			x[k] = predictors[k].normalize_vec(x[k])
			#終値ベクトルをRNNに投入できるように変形する
			x_rnn[k] = predictors[k].to_RNNvector(x[k])

		for i in ['4H']:
			#事前に予測した結果を保存しておく
			predictors[i].preset_prediction(x_rnn[i])

			#評価用にあらかじめ予測結果を取り出しておく
			#予測結果が存在している場合は計算をスキップできる
			if predictors[i].is_loaded() is True:
				predicted[i] = predictors[i].predicted
			else:
				predicted[i] = predictors[i].rnn.predict(candlesticks[i].ohlc)
			
		#特定の時間足のトレンドorレンジを判定
		is_trend = predictors['4H'].is_trend(x['4H'])

		#評価関数に注文情報をセット
		evaluator.begin_plotter()
		evaluator.add_candlestick(candlesticks)
		evaluator.add_predicted(predicted)
		evaluator.add_line(x)
		evaluator.end_plotter('signal.png', True)

		#evaluator.set_now_config(x,predicted,candlesticks)
		#LINE notification
		#evaluator.output_now()

		if True is is_trend:
			print('predictor judged TREND')

			#ひとつの時間足のトレンドを考慮する場合
			#特定の時間足の上昇or下降を予測、判定する
			is_rise = predictors['4H'].is_rise(x['4H'])

			#各時間足に対する条件（足かせ）をクリアするかを判定
			fetter_signals = {
				'15min': fetters['15min'].is_through(x['15min'], is_rise),
				'4H': fetters['4H'].is_through(x['4H'],is_rise)
			}

			#ひとつの時間足のトレンドを考慮する場合
			if all_element_in(fetter_signals) is True:
				print('All fetters clear')
				if is_rise is True:
					order_kind = 'BUY'
					pass
				else:
					order_kind = 'SELL'
					pass
				print(order_kind)
				is_order_created = trader.test_create_order(is_rise)

				#評価関数に注文情報をセット
				evaluator.set_order(order_kind)

				#evaluator.set_order_config(x,predicted,candlesticks,order_kind)
				#LINE notification
				#evaluator.output_order()

				if True is is_order_created:
					#ORDER状態からORDERWAITINGに状態遷移
					trader.switch_state()
				else:
					print('Some fetters do not clear')
					pass

		else:
			print('predictor judged RANGE')

		for k, v in candlesticks.items():
			predictors[k].reset()
	### Algorythm end ###
							

def	entry_antlia(key, candlesticks, trader, evaluator):
	### Algorythm begin ###
	if key == '1min' and trader.state == 'ORDER':
		hl_slopes = {}
		hl_intercepts = {}
		#安値トレンドライン
		debug_print('calculate Bottom trend line')
		slopes, intercepts = calc_trendline(candlesticks, 'low')
		hl_slopes['low'] = slopes
		hl_intercepts['low'] = intercepts
		#evaluator.output_trendline(candlesticks, slopes, intercepts, 'low')

		#高値トレンドライン
		debug_print('calculate Top trend line')
		slopes, intercepts = calc_trendline(candlesticks, 'high')
		hl_slopes['high'] = slopes
		hl_intercepts['high'] = intercepts

		#evaluator.output_trendline(candlesticks, slopes, intercepts, 'high')
		#evaluator.output_double_trendline(candlesticks, hl_slopes, hl_intercepts)

		print('############ Antlia Entry Strategy ############')
		#Ichimatsu Strategy
		slopes = {}
		intercepts = {}
		num_sets = {}
		ohlc = candlesticks['5min'].ohlc.copy()
		ohlc = (ohlc - ohlc.values.min()) / (ohlc.values.max() - ohlc.values.min())

		#ローソク足を2本1セットとして、numに対となるローソク足の本数を指定
		num_sets['5min'] = 4
		tail = ohlc[-num_sets['5min']:]

		close_open = ['close' if i%2 == 0 else 'open' for i in range(num_sets['5min'])]
		x = np.array([i + 1 for i in range(num_sets['5min'])])
		y = [tail[co].values[i] for i, co in enumerate(close_open)]
		#meno stderrを使うかもしれない
		slopes['5min'], intercepts['5min'], _, _, _ = linregress(x, y)
		#evaluator.output_tail(candlesticks['5min'].ohlc, slopes['5min'], intercepts['5min'], num_set)

		bottom = [+1 if i%2 == 0 else -1 for i in range(num_sets['5min'])]
		top    = [+1 if i%2 != 0 else -1 for i in range(num_sets['5min'])]
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
			evaluator.set_order(kind)

			#evaluator.output_ichimatsu(candlesticks, slopes, intercepts, num_sets)
			evaluator.begin_plotter()
			#evaluator.add_double_trendline(candlesticks, hl_slopes, hl_intercepts)
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
			#evaluator.output_ichimatsu(candlesticks, slopes, intercepts, num_sets)
			evaluator.begin_plotter()
			#evaluator.add_double_trendline(candlesticks, hl_slopes, hl_intercepts)
			evaluator.add_candlestick(candlesticks)
			evaluator.add_tail_oc_slope(candlesticks, slopes, intercepts, num_sets)
			evaluator.add_ichimatsu(candlesticks, num_sets)
			evaluator.end_plotter('signal.png', False)
			notify_from_line('progress', image='signal.png')

	### Algorythm end ###

def	entry_apus(key, candlesticks, trader, evaluator, predictors, fetters):
	### Algorythm begin ###
	if k == '5min' and trader.state == 'ORDER':

		x = {}
		x_rnn = {}
		predicted = {}

		
		#特定の時間足において予測を行う場合
		for k, v in candlesticks.items():
			#OHLCの中から終値をベクトルとして取り出す
			x[k] = predictors[k].from_ohlc_to_vector(v.ohlc, 'close')
			#取り出した終値ベクトルを正規化する
			x[k] = predictors[k].normalize_vec(x[k])
			#終値ベクトルをRNNに投入できるように変形する
			x_rnn[k] = predictors[k].to_RNNvector(x[k])

		for i in ['5min']:
			#事前に予測した結果を保存しておく
			predictors[i].preset_prediction(x_rnn[i])

			#評価用にあらかじめ予測結果を取り出しておく
			#予測結果が存在している場合は計算をスキップできる
			if predictors[i].is_loaded() is True:
				predicted[i] = predictors[i].predicted
			else:
				predicted[i] = predictors[i].rnn.predict(candlesticks[i].ohlc)
			
		#特定の時間足のトレンドorレンジを判定
		is_trend = predictors['5min'].is_trend(x['5min'])

		#現在の状況を可視化
		#evaluator.set_now_config(x,predicted,candlesticks)
		#evaluator.output_now()
		evaluator.begin_plotter()
		evaluator.add_candlestick(candlesticks)
		evaluator.add_predicted(x, predicted)
		evaluator.add_line(x)
		evaluator.end_plotter('signal.png', True)

		if True is is_trend:
			print('predictor judged TREND')

			#ひとつの時間足のトレンドを考慮する場合
			#特定の時間足の上昇or下降を予測、判定する
			is_rise = predictors['5min'].is_rise(x['5min'])
			"""
			#関数の実装がまだなのでダミーでTrueを
			is_rise = True
			"""

			#各時間足に対する条件（足かせ）をクリアするかを判定
			fetter_signals = {
				'1min': fetters['1min'].is_through(x['1min'],is_rise),
				'5min': fetters['5min'].is_through(x['5min'],is_rise)
			}

			#ひとつの時間足のトレンドを考慮する場合
			if all_element_in(fetter_signals) is True:
				print('All fetters clear')
				if is_rise is True:
					order_kind = 'BUY'
					pass
				else:
					order_kind = 'SELL'
					pass
				print(order_kind)
				is_order_created = trader.test_create_order(is_rise)

				#評価関数に注文情報をセット
				#evaluator.set_order_config(x,predicted,candlesticks,order_kind)
				#evaluator.output_order()
				evaluator.set_order(order_kind)

				if True is is_order_created:
					#ORDER状態からORDERWAITINGに状態遷移
					trader.switch_state()
				else:
					print('Some fetters do not clear')
					pass

		else:
			print('predictor judged RANGE')

		for k, v in candlesticks.items():
			predictors[k].reset()
	### Algorythm end ###

def entry_aquarius(key, candlesticks, trader, evaluator, predictors, fetters):
	### Algorythm begin ###
	if k == '15min' and trader.state == 'ORDER':
		x = {}
		x_rnn = {}
		predicted = {}
		
		#特定の時間足において予測を行う場合
		for i in range(['4H']):
			#OHLCの中から終値をベクトルとして取り出す
			x[i] = predictors[i].from_ohlc_to_vector(candlesticks[i].ohlc, 'close')
			#取り出した終値ベクトルを正規化する
			x[i] = predictors[i].normalize_vec(x[i])
			#終値ベクトルをRNNに投入できるように変形する
			x_rnn[i] = predictors[i].to_RNNvector(x[i])

			#事前に予測した結果を保存しておく
			predictors[i].preset_prediction(x_rnn[i])

			#評価用にあらかじめ予測結果を取り出しておく
			#予測結果が存在している場合は計算をスキップできる
			if predictors[i].is_loaded() is True:
				predicted[i] = predictors[i].predicted
			else:
				predicted[i] = predictors[i].rnn.predict(candlesticks[i].ohlc)

		#すべての時間足において予測を行う場合
		for k, v in candlesticks.items():
			#OHLCの中から終値をベクトルとして取り出す
			x[k] = predictors[k].from_ohlc_to_vector(v.ohlc, 'close')
			#取り出した終値ベクトルを正規化する
			x[k] = predictors[k].normalize_vec(x[k])
			#終値ベクトルをRNNに投入できるように変形する
			x_rnn[k] = predictors[k].to_RNNvector(x[k])

			#事前に予測した結果を保存しておく
			predictors[k].preset_prediction(x_rnn[k])

			#評価用にあらかじめ予測結果を取り出しておく
			#予測結果が存在している場合は計算をスキップできる
			if predictors[k].is_loaded() is True:
				predicted[k] = predictors[k].predicted
			else:
				predicted[k] = predictors[k].rnn.predict(v.ohlc)

		#特定の時間足のトレンドorレンジを考慮する場合
		trend_signals = {
			'4H': predictors['4H'].is_trend(candlesticks['4H'].ohlc)
		}
		#すべての時間足のトレンドorレンジを考慮する場合
		trend_signals = {	
			k: predictors[k].is_trend(v.ohlc) for k, v in candlesticks.items()
		}

		if all_element_in(trend_signals) is True:
			#特定の時間足の上昇or下降を考慮する場合
			rise_signals = {
				'4H': predictors['4H'].is_rise(candlesticks['4H'])
			}
			#複数の時間足の上昇or下降を考慮する場合
			rise_signals = {	
				k: predictors[k].is_rise(v.ohlc) for k, v in candlesticks.items()
			}
			is_rise = all_element_in(rise_signals)
			if is_rise is not None:
				#特定の時間足に対する条件（足かせ）をクリアするかを判定
				fetter_signals = {
					'4H': fetters['4H'].is_through(candlesticks['4H'].ohlc,is_rise)
				}
				#すべての時間足に対する条件（足かせ）をクリアするかを判定
				fetter_signals = {
					k: fetters[k].is_through(v.ohlc, is_rise) for k, v in candlesticks.items()
				}

				#足かせをすべて通過するか
				if all_element_in(fetters) is True:
					if is_rise is True:
						order_kind = 'BUY'
						#指値や逆指値の設定
						pass
					elif is_rise is False:
						order_kind = 'SELL'
						#指値や逆指値の設定
						pass
					else:
						pass
					#注文を発行
					is_order_created = trader.test_create_order(is_rise)

					#評価関数に注文情報をセット
					evaluator.set_order_config(x,predicted,candlesticks,order_kind)
					#LINE notification
					evaluator.output_order()

					if True is is_order_created:
						#ORDER -> WAIT_ORDER
						trader.switch_state()
					else:
						debug_print('order can not be created')
						debug_print('it may be bug if the message appear')
						pass
				else:
					pass
			else:
				debug_print('Signal is not distinct.')
				debug_print('Order can not be created')
				pass

		for k, v in candlesticks.items():
			predictors[k].reset()
		else:
			print('predictor judged RANGE')
	### Algorythm end ###


def settle_apricot(key, candlesticks, trader, evaluator):
	### Algorythm begin ###
	if key == '1min' and trader.state == 'POSITION':
		print('############ Apricot Settle Strategy ############')
		#ポジションを決済可能か
		if trader.can_close_position() is True:
			#決済の注文を発行する
			is_position_closed = trader.test_close_position()
			
			#evaluator.set_close_config(correct, candlesticks)
			#evaluator.output_candlestick(candlesticks)
			evaluator.begin_plotter()
			evaluator.add_candlestick(candlesticks)
			evaluator.end_plotter('close.png', True)

			evaluator.set_close()
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


def settle_strawberry(key, candlesticks, trader, evaluator):
	if key == '5min' and trader.state == 'POSITION':
		#ポジションを決済可能か
		if trader.can_close_position() is True:
			#決済の注文を発行する
			is_position_closed = trader.test_close_position()

			x = {}
			correct = {}
			for k, v in candlesticks.items():
				#for evaluate
				#OHLCの中から終値をベクトルとして取り出す
				x[k] = predictors[k].from_ohlc_to_vector(v.ohlc, 'close')
				#取り出した終値ベクトルを正規化する
				x[k] = predictors[k].normalize_vec(x[k])

				#要修正
				threshold = 2
				correct[k] = np.mean(x[k][-threshold:])
			#evaluator.set_close()
			#evaluator.set_close_config(correct, candlesticks)
			#LINE notification
			#evaluator.output_candlestick()
			#evaluator.log_score()

			evaluator.begin_plotter()
			evaluator.add_candlestick(candlesticks)
			evaluator.end_plotter('close.png', True)

			evaluator.set_close()
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


def settle_persimmon(key, candlesticks, trader, evaluator):
	if key == '15min' and trader.state == 'POSITION':
		#ポジションを決済可能か
		if trader.can_close_position() is True:
			#決済の注文を発行する
			is_position_closed = trader.test_close_position()

			x = {}
			correct = {}
			for k, v in candlesticks.items():
				#for evaluate
				#OHLCの中から終値をベクトルとして取り出す
				x[k] = predictors[k].from_ohlc_to_vector(v.ohlc, 'close')
				#取り出した終値ベクトルを正規化する
				x[k] = predictors[k].normalize_vec(x[k])

				#要修正
				threshold = 2
				correct[k] = np.mean(x[k][-threshold:])
			evaluator.set_close_config(correct, candlesticks)

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
	"""
	timeframes = {
		'15min': RNNparam('15min').tau,
		'4H': RNNparam('4H').tau
	}
	"""
	timeframes = {
		'1min': RNNparam('1min').tau,
		'5min': RNNparam('5min').tau
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
