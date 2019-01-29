import requests
import json
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import linregress

from OandaEndpoints import Order, Position, Pricing, Instrument
from OandaCandleStick import CandleStick
from predict_RNN import RNNPredictor
from RNNparam import RNNparam
import mpl_finance as mpf

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
	routiner = Routine_inspector(freq=10)

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


class Fetter:
	def __init__(self, rate):
		self.rate = rate

	#Strategy 1 for 4H
	def foo(self, ohlc):
		#OHLCデータについて計算
		x = ohlc
		x = 1

		#計算後のデータを返却
		return x

	def is_foo(self, ohlc):
		if self.rate != '4H':
			return False

		hoge = self.foo(ohlc)
		#fooから帰ってきた値に対してTrue or Falseを判断
		if hoge > 0:
			return True
		else:
			return False

	#Strategy 2 for 4H
	def foofoo(self, ohlc):
		#OHLCデータについて計算
		x = ohlc
		x = 1

		#計算後のデータを返却
		return x

	def is_foofoo(self, ohlc):
		if self.rate != '4H':
			return False

		hoge = self.foofoo(ohlc)
		if hoge > 0:
			return True
		else:
			return False

	#Strategy 1 for 15min
	def bar(self, ohlc):
		#OHLCデータについて計算
		x = ohlc
		x = 1

		#計算後のデータを返却
		return x

	def is_bar(self, ohlc):
		if self.rate != '15min':
			return False

		hoge = self.bar(ohlc)
		if hoge > 0:
			return True
		else:
			return False

	#Strategy 2 for 15min
	def barbar(self, ohlc):
		#OHLCデータについて計算
		x = ohlc
		x = 1

		#計算後のデータを返却
		return x

	def is_barbar(self, ohlc):
		if self.rate != '15min':
			return False

		hoge = self.barbar(ohlc)
		if hoge > 0:
			return True
		else:
			return False

	#Strategy 2 for 15min
	def barbarbar(self, ohlc):
		#OHLCデータについて計算
		x = ohlc
		x = 1

		#計算後のデータを返却
		return x

	def is_barbarbar(self, ohlc):
		if self.rate != '1min':
			return False

		hoge = self.barbar(ohlc)
		if hoge > 0:
			return True
		else:
			return False

	#Strategy 2 for 15min
	def barbarbarbar(self, ohlc):
		#OHLCデータについて計算
		x = ohlc
		x = 1

		#計算後のデータを返却
		return x

	def is_barbarbarbar(self, ohlc):
		if self.rate != '5min':
			return False

		hoge = self.barbar(ohlc)
		if hoge > 0:
			return True
		else:
			return False

	def is_through(self, ohlc, is_rise):

		if self.rate == '4H':
			if self.is_foofoo(ohlc) is False:
				return False
			if self.is_foo(ohlc) is False:
				return False

			return True

		elif self.rate == '15min':
			if self.is_bar(ohlc) is False:
				return False
			if self.is_barbar(ohlc) is False:
				return False

			return True

		elif self.rate == '1min':
			if self.is_barbarbar(ohlc) is False:
				return False

			return True

		elif self.rate == '5min':
			if self.is_barbarbarbar(ohlc) is False:
				return False

			return True
		else:
			return False

class Routine_inspector:
	def __init__(self, freq=10):
		self.CHECKING_FREQ = freq
		self.count = 0

	def is_inspect(self):
		if self.count == self.CHECKING_FREQ:
			return True
		else:
			return False

	def is_inspect_at_intervals_of(self, freq):
		if self.count == freq:
			return True
		else:
			return False
	
	def increment_count(self):
		self.count += 1
		return self.count

	def reset_count(self):
		self.count = 0
		return self.count

	def update(self):
		if self.is_inspect() is True:
			self.reset_count()
			return True
		else:
			self.increment_count()
			return False

	def update_at_intervals_of(self, freq):
		if self.is_inspect_at_intervals_of(freq) is True:
			self.reset_count()
			return True
		else:
			self.increment_count()
			return False

class Plotter:
	def __init__(self, nrows, ncols, figsize):
		self.fig, self.axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize,squeeze=False)

	def plot(self, x, row, col=0):
		self.axes[row, col].plot(x, color='royalblue')

	def plot_predict(self, x, predicted, row, col=0):
		tmp = np.append(x, predicted)
		self.axes[row,col].plot(tmp, color='red', linestyle='dashed')

	def plot_correct(self, x, correct, row, col=0):
		tmp = np.append(x, correct)
		self.axes[row,col].plot(tmp, color='royalblue')
		
	def plot_ohlc(self, ohlc, row, col=0):
		#ohlc = (ohlc - ohlc.min()) / (ohlc.max() - ohlc.min())
		ohlc = (ohlc - ohlc.values.min()) / (ohlc.values.max() - ohlc.values.min())
		time = ohlc.index[-1].strftime('%Y-%m-%d %H:%M:%S')
		iohlc = np.vstack((range(len(ohlc)), ohlc.values.T)).T

		self.axes[row,col].patch.set_facecolor('black')
		width = 0.7
		cu = 'black'
		cd = 'white'
		ls, rs = mpf.candlestick_ohlc(self.axes[row,col], iohlc, width=width, colorup=cu, colordown=cd)
		for r in rs:
			r.set_edgecolor('green')
			r.set_linewidth(1.2)

	def savefig(self, path):
		self.fig.savefig(path)
	
	def close(self):
		plt.close()	

	def set_title(self, title, row, col=0):
		self.axes[row, col].set_title(title, fontsize=14)

	def set_ylim(self, y_min, y_max, row, col=0):
		self.axes[row, col].set_ylim(y_min, y_max)

	def set_for_score(self):
		pass

	def set_for_close(self):
		pass

	def set_for_order(self):
		pass

	def plot_tail_oc_slope(self, ohlc, slope, intercept, num_set, row, col=0):
		x = np.array([i + 1 for i in range(num_set)])
		y = slope * x + intercept
		self.axes[row,col].plot([len(ohlc) - i - 1 for i in reversed(range(num_set))],y[-num_set:])

	def plot_ichimatsu(self, ohlc, num_set, row, col=0):
		print(ohlc[-num_set:])
		print(ohlc[-num_set:].max())
		print(np.max(ohlc[-num_set:].max().values))
		print(ohlc[-num_set:].values.max())
		
		ohlc = (ohlc - ohlc.values.min()) / (ohlc.values.max() - ohlc.values.min())

		xy = (len(ohlc) - num_set - 0.5 - 0.1, ohlc[-num_set:].values.min())
		width = num_set + 0.2
		height = ohlc[-num_set:].values.max() - ohlc[-num_set:].values.min()
		ec='#990000'
		rect = patches.Rectangle(xy=xy, width=width, height=height, ec=ec, fill=False)
		print(rect)
		self.axes[row,col].add_patch(rect)

	def plot_trendline(self, ohlc, slope, intercept, row, col=0):
		ohlc = (ohlc - ohlc.min()) / (ohlc.max() - ohlc.min())
		#time = ohlc.index[-1].strftime('%Y-%m-%d %H:%M:%S')
		iohlc = np.vstack((range(len(ohlc)), ohlc.values.T)).T
		
		x = np.array([i+1 for i in range(len(ohlc))])
		y = slope * x + intercept
		self.axes[row,col].plot(y)

class Evaluator:
	def __init__(self, timeframes, instrument, environment='demo'):
		self.pre_price = None
		self.post_price = None
		self.profit = 0
		self.log = {
			'win_rate': [],
			'profit': [0],
			'win_or_lose': []
		}
		self.instrument = instrument
		self.pricing = Pricing(environment)

		self.kind = None

		#取り扱う時間足のリスト
		#list
		self.timeframes = list(timeframes)

		self.nrows = len(timeframes)
		self.ncols = 1
		self.figsize = (9, 3*self.nrows)

	def set_order(self, kind):
		resp = self.pricing.get_pricing_information(self.instrument)
		msg = from_byte_to_dict(resp.content)
		self.pre_price = float(msg['prices'][0]['bids'][0]['price'])
		self.kind = kind

	def set_close(self):
		resp = self.pricing.get_pricing_information(self.instrument)
		msg = from_byte_to_dict(resp.content)
		self.post_price = float(msg['prices'][0]['bids'][0]['price'])

	def begin_plotter(self):
		self.plotter = Plotter(nrows=self.nrows, ncols=self.ncols, figsize=self.figsize)

	def end_plotter(self, file_name, notify):
		for i, k in enumerate(self.timeframes):
			self.plotter.set_title(k, i)
		self.plotter.savefig(file_name)
		self.plotter.close()
		if notify is True:
			notify_from_line(self.kind, image=file_name)
			
	def add_tail_oc_slope(self, candlesticks, slopes, intercepts, num_sets):
		for i, k in enumerate(self.timeframes):
			if k in slopes:
				self.plotter.plot_tail_oc_slope(candlesticks[k].ohlc,slopes[k],intercepts[k],num_sets[k], i)

	def add_ichimatsu(self, candlesticks, num_sets):
		for i, k in enumerate(self.timeframes):
			if k in num_sets:
				self.plotter.plot_ichimatsu(candlesticks[k].ohlc, num_sets[k], i)

	def add_candlestick(self, candlesticks):
		for i, k in enumerate(self.timeframes):
			if k in candlesticks:
				self.plotter.plot_ohlc(candlesticks[k].ohlc, i)

	def add_trendline(self, candlesticks, slopes, intercepts):
		for i, k in enumerate(self.timeframes):
			if k in slopes:
				self.plotter.plot_trendline(candlesticks[k].ohlc,slopes[k],intercepts[k], i)

	def add_double_trendline(self, candlesticks, slopes, intercepts):
		for i, k in enumerate(self.timeframes):
			if k in slopes['high']:
				self.plotter.plot_trendline(candlesticks[k].ohlc,slopes['high'][k],intercepts['high'][k], i)
			if k in slopes['low']:
				self.plotter.plot_trendline(candlesticks[k].ohlc,slopes['low'][k],intercepts['low'][k], i)

	def add_predicted(self, x, predicted):
		for i, k in enumerate(self.timeframes):
			if k in self.predicted:
				self.plotter.plot_predict(x[k], predicted[k], i)

	def add_line(self, x):
		for i, k in enumerate(self.timeframes):
			if k in x_ordered:
				self.plotter.plot(x[k], i)

	def output_ichimatsu(self, candlesticks, slopes, intercepts, num_sets, notify=True):
		plotter = Plotter(nrows=self.nrows, ncols=self.ncols, figsize=self.figsize)
		file_name = 'ichimatsu.png'
		for i, k in enumerate(self.timeframes):
			if k in candlesticks:
				plotter.plot_ohlc(candlesticks[k].ohlc, i)
			if k in slopes:
				plotter.plot_ichimatsu(candlesticks[k].ohlc,slopes[k],intercepts[k],num_sets[k], i)

			#確実に設定できるものを記述
			plotter.set_title(k, i)
		#その他設定
		plotter.savefig(file_name)
		plotter.close()
		if notify is True:
			notify_from_line(self.kind, image=file_name)

	def output_trendline(self, candlesticks, slopes, intercepts, price='high', notify=True):
		plotter = Plotter(nrows=self.nrows, ncols=self.ncols, figsize=self.figsize)
		file_name = f'{price}_trendline.png'
		for i, k in enumerate(self.timeframes):
			if k in candlesticks:
				plotter.plot_ohlc(candlesticks[k].ohlc, i)
			if k in slopes:
				plotter.plot_trendline(candlesticks[k].ohlc,slopes[k],intercepts[k], i)

			#確実に設定できるものを記述
			plotter.set_title(k, i)
		#その他設定
		plotter.savefig(file_name)
		plotter.close()
		if notify is True:
			notify_from_line(self.kind, image=file_name)

	def output_double_trendline(self, candlesticks, slopes, intercepts, notify=True):
		plotter = Plotter(nrows=self.nrows, ncols=self.ncols, figsize=self.figsize)
		file_name = f'trendline.png'
		for i, k in enumerate(self.timeframes):
			if k in candlesticks:
				plotter.plot_ohlc(candlesticks[k].ohlc, i)
			if k in slopes['high']:
				plotter.plot_trendline(candlesticks[k].ohlc,slopes['high'][k],intercepts['high'][k], i)
			if k in slopes['low']:
				plotter.plot_trendline(candlesticks[k].ohlc,slopes['low'][k],intercepts['low'][k], i)

			#確実に設定できるものを記述
			plotter.set_title(k, i)
		#その他設定
		plotter.savefig(file_name)
		plotter.close()
		if notify is True:
			notify_from_line(self.kind, image=file_name)

	def output_candlestick(self, candlesticks, notify=True):
		plotter = Plotter(nrows=self.nrows, ncols=self.ncols, figsize=self.figsize)
		file_name = 'candlesticks.png'
		for i, k in enumerate(self.timeframes):
			if k in candlesticks:
				plotter.plot_ohlc(candlesticks[k].ohlc, i)

			#確実に設定できるものを記述
			plotter.set_title(k, i)
			#self.axes[i,0].set_title(k, fontsize=18)
		#その他設定
		plotter.set_for_close()
		plotter.savefig(file_name)
		plotter.close()
		if notify is True:
			notify_from_line('close', image=file_name)

	def output_score(self, msg, notify=True):
		plotter = Plotter(nrows=2, ncols=1, figsize=(8,8))
		file_name = 'score.png'
		for i, key in enumerate(['profit', 'win_rate']):
			plotter.plot(self.log[key], i)

			plotter.set_title(key, i)
			#あとでset_for_scoreにつっこむ
			if i == 1:
				plotter.set_ylim(-0.25,1.25, i)
		#その他設定
		plotter.set_for_score()
		plotter.savefig(file_name)
		plotter.close()
		if notify is True:
			notify_from_line(msg, image=file_name)

	def log_score(self):
		diff = self.post_price - self.pre_price
		diff = diff if self.kind == 'BUY' else -diff
		self.log['profit'].append(self.log['profit'][-1] + diff)

		sign = np.sign(self.post_price - self.pre_price)
		sign = sign if self.kind == 'BUY' else -sign
		self.log['win_or_lose'].append(sign)

		msg = self.kind
		msg += ' WIN' if self.log['win_or_lose'][-1] > 0 else ' LOSE'

		print(f'kind: {self.kind} raw_sign: {sign}  diff: {diff}')
		print(f'profit: {self.log["profit"][-1]} {self.log["profit"][-1] + diff}')

		print(msg)	

		win_rate = self.log['win_or_lose'].count(1) / len(self.log['win_or_lose'])
		self.log['win_rate'].append(win_rate)
		print(f'win_rate: {win_rate}')

		self.output_score(msg)

class Trader:
	def __init__(self, instrument, environment='demo', mode='test'):
		self.state = None
		self.mode = mode
		self.instrument = instrument
		self.order = Order(environment)
		self.position = Position(environment)

		#for debug
		self.count = 0

	def arrange_data_for_limit(self, kind, instrument, units, price):
		price = '{}'.format(price)
		sign = '+' if kind is 'BUY' else '-'
		units = '{}{}'.format(sign, units)
		data = {
			'order': {
				'price': price,
				'instrument': instrument,
				'units':units,
				'type': 'LIMIT',
				'positionFill': 'DEFAULT'
			}
		}
		return data

	def arrange_data_for_market(self, kind, instrument, units):
		sign = '+' if kind is 'BUY' else '-'
		units = '{}{}'.format(sign, units)
		data = {
			'order': {
				'instrument': instrument,
				'units':units,
				'type': 'MARKET',
				'positionFill': 'DEFAULT'
			}
		}
		return data

	def test_create_order(self, is_rise=None):
		if is_rise is True:
			debug_print('BUY')
		elif is_rise is False:
			debug_print('SELL')
		else:
			return False
				
		return True

	def create_order(self, is_rise=None):
		if is_rise is True:
			if self.mode == 'test':
				debug_print('BUY')
			else:
				self.market_order('BUY', self.instrument, 10)
				time.sleep(1)
				print('create order BUY')
		elif is_rise is False:
			if self.mode == 'test':
				debug_print('SELL')
			else:
				self.market_order('SELL', self.instrument, 10)
				time.sleep(1)
				print('create order SELL')
		else:
			return False
				
		return True

	def test_close_position(self):
		#for debug
		debug_print('Close position')
		return True

	def close_position(self):
		if self.mode == 'test':
			pass
		else:
			resp = self.position.get_open_position()
			time.sleep(1)
			pos_dic = from_response_to_dict(resp)
			for i in range(len(pos_dic['positions'])):
				instrument = pos_dic['positions'][i]['instrument']
				for order_type in ['long','short']:
					if pos_dic['positions'][i][order_type]['units'] is not '0':
						data = {
							'{}Units'.format(order_type): 'ALL'
						}
						self.position.close_position(data, instrument)
						time.sleep(1)

		print('Close position')
		return True

	def can_close_position(self):
		threshold = 1
		if self.count > threshold:
			return True
		else:
			return False

	def update_whether_closing_position(self):
		if self.can_close_position() is True:
			self.count = 0
		else:
			self.count += 1
	
	def test_is_reflected_order(self):
		if self.state == 'WAIT_ORDER':
			resp = self.order.get_pending_list()
			order_dic = from_response_to_dict(resp)
			time.sleep(1)
			
			#empty order for debug
			order_dic = {
				'orders': []
			}

			if not self.has_order(order_dic) is True:
				return True
		return False

	def is_reflected_order(self):
		if self.state == 'WAIT_ORDER':
			resp = self.order.get_pending_list()
			order_dic = from_response_to_dict(resp)
			time.sleep(1)
			
			if self.mode == 'test':
				#empty order for debug
				order_dic = {
					'orders': []
				}
			else:
				pass

			if not self.has_order(order_dic) is True:
				return True
		return False

	def test_is_reflected_position(self):
		if self.state == 'WAIT_POSITION':
			resp = self.position.get_open_position()
			pos_dic = from_response_to_dict(resp)
			time.sleep(1)

			#empty position for debug
			pos_dic = {
				'positions': []
			}

			if not self.has_position(pos_dic) is True:
				return True
		return False

	def is_reflected_position(self):
		if self.state == 'WAIT_POSITION':
			resp = self.position.get_open_position()
			pos_dic = from_response_to_dict(resp)
			time.sleep(1)

			if self.mode == 'test':
				#empty position for debug
				pos_dic = {
					'positions': []
				}
			else:
				pass

			if not self.has_position(pos_dic) is True:
				return True
		return False

	#保留中の注文と保有中のポジションを空にする
	def clean(self):
		resp = self.order.get_pending_list()
		time.sleep(1)
		order_dic = from_response_to_dict(resp)

		resp = self.position.get_open_position()
		time.sleep(1)
		pos_dic = from_response_to_dict(resp)

		
		#保留中の注文をすべてキャンセルする
		if self.has_order(order_dic) is True:
			for i in range(len(order_dic['orders'])):
				order_id = order_dic['orders'][i]['id']
				self.order.cancel_order(order_id)
				time.sleep(1)
			debug_print('cancel order')

		#Oandaのサーバー上に反映されるまで待つ
		while self.has_order(order_dic) is True:
			resp = self.order.get_pending_list()
			time.sleep(1)
			order_dic = from_response_to_dict(resp)
			has_odr = 'exist' if self.has_order(order_dic) is True else 'clean'
			print('pending order: {}'.format(has_odr))

		#保有しているポジションをすべて決済する
		if self.has_position(pos_dic) is True:
			for i in range(len(pos_dic['positions'])):
				instrument = pos_dic['positions'][i]['instrument']
				for order_type in ['long','short']:
					if pos_dic['positions'][i][order_type]['units'] is not '0':
						data = {
							'{}Units'.format(order_type): 'ALL'
						}
						self.position.close_position(data, instrument)
						time.sleep(1)
			debug_print('close positon')

		#Oandaのサーバー上に反映されるまで待つ
		while self.has_position(pos_dic) is True:
			resp = self.position.get_open_position()
			time.sleep(1)
			pos_dic = from_response_to_dict(resp)
			has_pos = 'exist' if self.has_position(pos_dic) is True else 'clean'
			print('position: {}'.format(has_pos))

		#ORDERモードに状態遷移する
		self.state = 'ORDER'
		print(self.state)
		return self.state

	def execute(self):
		if self.state == 'ORDER':
			has_ord = self.exec_order()
			#+limit order
			#+stop order
			if has_ord is True:
				self.switch_state()

		elif self.state == 'POSITION':
			has_pos = self.exec_position()
			if has_pos is True:
				self.switch_state()

		else:
			pass
			
		if self.is_reflected_order() is True:
			self.switch_state()
		if self.is_reflected_position() is True:
			self.switch_state()
		print(self.state)
		return self.state

	def exec_order(self, is_rise=None):
		if is_rise is True:
			self.market_order('BUY', self.instrument, 10)
			time.sleep(1)
			print('create order BUY')

			#for evaluate
			order_kind = 'Buy'
		else:
			self.market_order('SELL', self.instrument, 10)
			time.sleep(1)
			print('create order SELL')

			#for evaluate
			order_kind = 'Sell'

		return order_kind

	def exec_position(self):
		self.counter += 1
		threshold = 4
		print(f'counter: {self.counter} threshold: {threshold}')
		if self.counter > threshold:
			resp = self.position.get_open_position()
			time.sleep(1)
			pos_dic = from_response_to_dict(resp)
			for i in range(len(pos_dic['positions'])):
				instrument = pos_dic['positions'][i]['instrument']
				for order_type in ['long','short']:
					if pos_dic['positions'][i][order_type]['units'] is not '0':
						data = {
							'{}Units'.format(order_type): 'ALL'
						}
						self.position.close_position(data, instrument)
						time.sleep(1)

			print('close position')
			#reset conter
			self.counter = 0

			return True
		else:
			return False

	def has_order(self, order_dic):
		if bool(order_dic['orders']) is True:
			return True
		else:
			return False

	def has_position(self, pos_dic):
		if bool(pos_dic['positions']) is True:
			return True
		else:
			return False

	#指値注文
	def limit_order(self, kind, instrument, units, price):
		data = self.arrange_data_for_limit(kind, instrument, units, price)					
		resp = self.order.create_order(data)
		#for debug
		print(resp.text)

		#stateは変化させない
		#self.state = 'WAIT_ORDER'

	#成行注文
	def market_order(self, kind, instrument, units):
		data = self.arrange_data_for_market(kind, instrument, units)					
		resp = self.order.create_order(data)
		#for debug
		print(resp.text)

		#stateは変化させない
		#self.state = 'WAIT_ORDER'

	def switch_state(self):
		if self.state == 'ORDER':
			self.state = 'WAIT_ORDER'

		elif self.state == 'WAIT_ORDER':
			self.state = 'POSITION'

		elif self.state == 'POSITION':
			self.state = 'WAIT_POSITION'

		elif self.state == 'WAIT_POSITION':
			self.state = 'ORDER'
		
		else:
			pass

class Predictor:
	def __init__(self, rate):
		param = RNNparam(rate)
		#Layer config
		input_dim = param.input_dim
		output_dim = param.output_dim
		hidden_dims = param.hidden_dims #hidden_dims is stored in list for future expansion
		actvs = param.actvs # actvs may store activation function for each cell 

		tau = param.tau
		hidden_units = param.hidden_units
		keep_prob = param.keep_prob

		#if rate == '15min':
			#Cell config -> RNN Class member variable
		#	tau = 60
		#	hidden_units = hidden_dims[0]
		#	keep_prob = 1.0

		#	candle_info = {
		#		'preproc': 'norm',
		#		'x_form': None,
		#		't_form': ['average', 36], #t_form: ['shift', 0]
		#		'rate' : '15min',
		#		'price' : 'close',
		#		'tau' : tau,
		#	}

		#if rate == '4H':
			#Cell config -> RNN Class member variable
		#	tau = 45
		#	hidden_units = hidden_dims[0]
		#	keep_prob = 1.0

		#	candle_info = {
		#		'preproc': 'norm',
		#		'x_form': None,
		#		't_form': ['average', 15], #t_form: ['shift', 0]
		#		'rate' : '4H',
		#		'price' : 'close',
		#		'tau' : tau,
		#	}

		self.rnn = RNNPredictor(rate=rate, load=True, save=False)
		self.rnn.set_config(input_dim, output_dim, hidden_dims, actvs, tau, keep_prob)
		self.rnn.preload_model()

		#reset by every routine
		self.predicted = None
		self.loaded = False

		self.x_max = None
		self.x_min = None
		self.is_normalized = False

	def reset(self):
		self.predicted = None
		self.loaded = False

	def to_RNNvector(self, x):
		x_rnn = np.array([[x]]).transpose(0,2,1)
		return x_rnn

	def from_ohlc_to_vector(self, candlestick, key='close'):
		x = candlestick[key].values
		return x

	def set_min_max_of(self, x):
		self.x_min = x.min()
		self.x_max = x.max()

	def clear_normalize_config(self):
		self.x_max = None
		self.x_min = None
		self.is_normalized = False

	def normalize_vec(self, x):
		self.is_normalized = True
		self.set_min_max_of(x)
		return (x - x.min()) / (x.max() - x.min())

	def denormalize_vec(self, x):
		if is_normalized is True:
			return x * (self.x_max - self.x_min) + self.x_min
		else:
			return x

	def is_fall(self, candlestick):
		#predicted = self.predicted if loaded is True else self.rnn.predict(candlestick)
		if self.loaded is True:
			predicted = self.predicted
		else:
			predicted = self.rnn.predict(candlestick)

		if np.sign(predicted - candlestick[-1]) < 0:
			return True
		else:
			return False

	def is_range(self, candlestick):
		#predicted = self.predicted if loaded is True else self.rnn.predict(candlestick)

		if self.loaded is True:
			predicted = self.predicted
		else:
			predicted = self.rnn.predict(candlestick)

		threshold = 0.2
		diff = np.abs(predicted - candlestick[-1])
		print('range {}'.format(diff[0,0]))
		if diff[0,0] < threshold:
			print('is_range True')
			return True
		else:
			print('is_range false')
			return False

	def is_rise(self, candlestick):
		#predicted = self.predicted if loaded is True else self.rnn.predict(candlestick)

		if self.loaded is True:
			predicted = self.predicted
		else:
			predicted = self.rnn.predict(candlestick)

		if np.sign(predicted - candlestick[-1]) > 0:
			return True
		else:
			return False

	def is_trend(self, candlestick):
		#predicted = self.predicted if loaded is True else self.rnn.predict(candlestick)

		if self.loaded is True:
			predicted = self.predicted
		else:
			predicted = self.rnn.predict(candlestick)

		debug_print(predicted)
		debug_print(candlestick[-1])

		threshold = 0.1
		diff = np.abs(predicted - candlestick[-1])
		print('is_trend: {}'.format(diff[0,0]))
		if diff[0,0] > threshold:
			print('trend')
			return True
		else:
			print('not trend')
			return False

	def preset_prediction(self, x):
		self.predicted = self.rnn.predict(x)
		self.loaded = True
		print('preset prediction')
		return self.predicted

	def is_loaded(self):
		if self.loaded is True:
			return True
		else:
			return False


class Strategy:
	def __init__(self, environment, instrument):
		self._state = None
		self.instrument = instrument
		self.order = Order(environment)
		self.position = Position(environment)

		self.evaluator = Evaluator(environment, instrument)
		self.has_evaluator = True

		#Layer config
		input_dim = 1
		output_dim = 1
		hidden_dims = [4] #hidden_dims is stored in list for future expansion
		actvs = ['tanh'] # actvs may store activation function for each cell 

		#Cell config -> RNN Class member variable
		tau = 12
		hidden_units = hidden_dims[0]
		keep_prob = 1.0

		candle_info = {
			'preproc': 'norm',
			'x_form': None,
			't_form': ['average', 12], #t_form: ['shift', 0]
			'rate' : '1min',
			'price' : 'close',
			'tau' : tau,
		}

		self.rnn = RNNPredictor(load=True, save=False)
		self.rnn.set_config(input_dim, output_dim, hidden_dims, actvs, tau, keep_prob)
		self.rnn.preload_model()

		#for debug
		self.counter = 0

	def arrange_data_for_limit(self, kind, instrument, units, price):
		price = '{}'.format(price)
		sign = '+' if kind is 'BUY' else '-'
		units = '{}{}'.format(sign, units)
		data = {
			'order': {
				'price': price,
				'instrument': instrument,
				'units':units,
				'type': 'LIMIT',
				'positionFill': 'DEFAULT'
			}
		}
		return data

	def arrange_data_for_market(self, kind, instrument, units):
		sign = '+' if kind is 'BUY' else '-'
		units = '{}{}'.format(sign, units)
		data = {
			'order': {
				'instrument': instrument,
				'units':units,
				'type': 'MARKET',
				'positionFill': 'DEFAULT'
			}
		}
		return data

	#bar... is used in test mode.
	def bar_execute(self, candlestick):
		if self.state == 'ORDER':
			has_ord = self.bar_exec_order(candlestick)
			#+limit order
			#+market order
			if has_ord is True:
				self.switch_state()

		elif self.state == 'POSITION':
			has_pos = self.bar_exec_position(candlestick)
			if has_pos is True:
				self.switch_state()

		else:
			pass
			
		if self.bar_is_reflected_order() is True:
			self.switch_state()
		if self.bar_is_reflected_position() is True:
			self.switch_state()
		print(self.state)
		return self.state


	def bar_exec_order(self, candlestick):
		x, x_rnn = self.convert_candlestick_into_vec(candlestick)

		is_loaded = self.preset_prediction(x_rnn, load=True)
		if self.is_range(x, loaded=is_loaded) is False:
			if self.is_rise(x, loaded=is_loaded) is True:
				#for evaluate
				debug_print('buy')
				order_kind = 'buy'
			else:
				#for evaluate
				debug_print('sell')
				order_kind = 'sell'

			#for evaluate
			if self.has_evaluator is True:
				predicted = self.predicted if is_loaded is True else self.rnn.predict(candlestick)
				self.evaluator.set_order_info(x, predicted, order_kind)

				#LINE notification
				self.evaluator.notify_order()

			return True
		else:
			debug_print('range')
			return False

	def bar_exec_position(self, candlestick):
		self.counter += 1
		threshold = 4
		print(f'conter: {self.counter} threshold: {threshold}')

		if self.counter > threshold:
			'''
			###
			#現在保持しているポジションを取得(position.get_open_position())
			#取得したデータをdictionaryに変換(from_response_to_dict())
			#保有しているポジションが0かどうかを確認して、クエリを生成
			#注文しているロング、ショート両方のポジションを決済(position.close_position())
			###
			resp = self.position.get_open_position()
			time.sleep(1)
			pos_dic = self.from_response_to_dict(resp)
			for i in range(len(pos_dic['positions'])):
				instrument = pos_dic['positions'][i]['instrument']
				for order_type in ['long','short']:
					if pos_dic['positions'][i][order_type]['units'] is not '0':
						data = {
							'{}Units'.format(order_type): 'ALL'
						}
						self.position.close_position(data, instrument)
						time.sleep(1)
			'''
			#for debug
			print('close pos')
			self.counter = 0

			#for evaluate
			if self.has_evaluator is True:
				x, x_rnn = self.convert_candlestick_into_vec(candlestick)
				correct = np.mean(x[-threshold:])
				self.evaluator.set_close_info(correct)

				#LINE notification
				self.evaluator.notify_close()
				self.evaluator.log_score()

			return True
		else:
			return False
	
	def bar_is_reflected_order(self):
		if self.state == 'WAIT_ORDER':
			resp = self.order.get_pending_list()
			order_dic = from_response_to_dict(resp)
			time.sleep(1)
			
			#empty order for debug
			order_dic = {
				'orders': []
			}
			if not self.has_order(order_dic) is True:
				return True
		return False

	def bar_is_reflected_position(self):
		if self.state == 'WAIT_POSITION':
			resp = self.position.get_open_position()
			pos_dic = from_response_to_dict(resp)
			time.sleep(1)

			#empty order for debug
			pos_dic = {
				'positions': []
			}
			if not self.has_position(pos_dic) is True:
				return True
		return False

	#保留中の注文と保有中のポジションを空にする
	def clean(self):
		resp = self.order.get_pending_list()
		time.sleep(1)
		order_dic = from_response_to_dict(resp)

		resp = self.position.get_open_position()
		time.sleep(1)
		pos_dic = from_response_to_dict(resp)

		
		#保留中の注文をすべてキャンセルする
		if self.has_order(order_dic) is True:
			for i in range(len(order_dic['orders'])):
				order_id = order_dic['orders'][i]['id']
				self.order.cancel_order(order_id)
				time.sleep(1)
			debug_print('cancel order')

		#Oandaのサーバー上に反映されるまで待つ
		while self.has_order(order_dic) is True:
			resp = self.order.get_pending_list()
			time.sleep(1)
			order_dic = from_response_to_dict(resp)
			has_odr = 'exist' if self.has_order(order_dic) is True else 'clean'
			print('pending order: {}'.format(has_odr))

		#保有しているポジションをすべて決済する
		if self.has_position(pos_dic) is True:
			for i in range(len(pos_dic['positions'])):
				instrument = pos_dic['positions'][i]['instrument']
				for order_type in ['long','short']:
					if pos_dic['positions'][i][order_type]['units'] is not '0':
						data = {
							'{}Units'.format(order_type): 'ALL'
						}
						self.position.close_position(data, instrument)
						time.sleep(1)
			debug_print('close positon')

		#Oandaのサーバー上に反映されるまで待つ
		while self.has_position(pos_dic) is True:
			resp = self.position.get_open_position()
			time.sleep(1)
			pos_dic = from_response_to_dict(resp)
			has_pos = 'exist' if self.has_position(pos_dic) is True else 'clean'
			print('position: {}'.format(has_pos))

		#ORDERモードに状態遷移する
		self.state = 'ORDER'
		print(self.state)
		return self.state

	def convert_candlestick_into_vec(self, candlestick):
		print(candlestick.ohlc['close'].values)
		print(candlestick.ohlc['close'].values.shape)
		x = candlestick.ohlc['close'].values

		self.x_min = x.min()
		self.x_max = x.max()

		x = (x - x.min()) / (x.max() - x.min())
		x_rnn = np.array([[x]]).transpose(0,2,1)

		return (x, x_rnn)

	def denormalize_vec(self, x):
		return x * (self.x_max - self.x_min) + self.x_min

	def execute(self):
		if self.state == 'ORDER':
			has_ord = self.exec_order(candlestick)
			#+limit order
			#+market order
			if has_ord is True:
				self.switch_state()

		elif self.state == 'POSITION':
			has_pos = self.exec_position(candlestick)
			if has_pos is True:
				self.switch_state()

		else:
			pass
			
		if self.is_reflected_order() is True:
			self.switch_state()
		if self.is_reflected_position() is True:
			self.switch_state()
		print(self.state)
		return self.state

	def exec_order(self):
		x, x_rnn = self.convert_candlestick_into_vec(candlestick)

		is_loaded = self.preset_prediction(x_rnn, load=True)
		if self.is_range(x, loaded=is_loaded) is False:
			if self.is_rise(x, loaded=is_loaded) is True:
				self.market_order('BUY', self.instrument, 10)
				time.sleep(1)
				print('create order BUY')

				#for evaluate
				order_kind = 'buy'
			else:
				self.market_order('SELL', self.instrument, 10)
				time.sleep(1)
				print('create order SELL')

				#for evaluate
				order_kind = 'sell'

			#for evaluate
			if self.has_evaluator is True:
				predicted = self.predicted if is_loaded is True else self.rnn.predict(candlestick)
				self.evaluator.set_order_info(x, predicted, order_kind)

				#LINE notification
				self.evaluator.notify_order()

			return True
		else:
			debug_print('range')
			return False

	def exec_position(self):
		self.counter += 1
		threshold = 4
		print(f'counter: {self.counter} threshold: {threshold}')
		if self.counter > threshold:
			resp = self.position.get_open_position()
			time.sleep(1)
			pos_dic = from_response_to_dict(resp)
			for i in range(len(pos_dic['positions'])):
				instrument = pos_dic['positions'][i]['instrument']
				for order_type in ['long','short']:
					if pos_dic['positions'][i][order_type]['units'] is not '0':
						data = {
							'{}Units'.format(order_type): 'ALL'
						}
						self.position.close_position(data, instrument)
						time.sleep(1)

			print('close position')
			#reset conter
			self.counter = 0

			#for evaluate
			if self.has_evaluator is True:
				x, x_rnn = self.convert_candlestick_into_vec(candlestick)
				correct = np.mean(x[-threshold:])
				self.evaluator.set_close_info(correct)

				#LINE notification
				self.evaluator.notify_close()
				self.evaluator.log_score()

			return True
		else:
			return False

#	def from_response_to_dict(self, response):
#		return json.loads(response.content.decode('UTF-8'))

	def has_order(self, order_dic):
		if bool(order_dic['orders']) is True:
			return True
		else:
			return False

	def has_position(self, pos_dic):
		if bool(pos_dic['positions']) is True:
			return True
		else:
			return False

	def is_fall(self, candlestick, loaded=False):
		predicted = self.predicted if loaded is True else self.rnn.predict(candlestick)

		if np.sign(predicted - candlestick[-1]) < 0:
			return True
		else:
			return False

	def is_range(self, candlestick, loaded=False):
		predicted = self.predicted if loaded is True else self.rnn.predict(candlestick)

		threshold = 0.2
		diff = np.abs(predicted - candlestick[-1])
		print('range {}'.format(diff[0,0]))
		if diff[0,0] < threshold:
			print('is_range True')
			return True
		else:
			print('is_range false')
			return False

	def is_reflected_order(self):
		if self.state == 'WAIT_ORDER':
			resp = self.order.get_pending_list()
			order_dic = from_response_to_dict(resp)
			time.sleep(1)
			
			if not self.has_order(order_dic) is True:
				return True
		return False

	def is_reflected_position(self):
		if self.state == 'WAIT_POSITION':
			resp = self.position.get_open_position()
			pos_dic = from_response_to_dict(resp)
			time.sleep(1)

			if not self.has_position(pos_dic) is True:
				return True
		return False

	def is_rise(self, candlestick, loaded=False):
		predicted = self.predicted if loaded is True else self.rnn.predict(candlestick)

		if np.sign(predicted - candlestick[-1]) > 0:
			return True
		else:
			return False

	#指値注文
	def limit_order(self, kind, instrument, units, price):
		data = self.arrange_data_for_limit(kind, instrument, units, price)					
		resp = self.order.create_order(data)
		#for debug
		print(resp.text)

		#stateは変化させない
		#self.state = 'WAIT_ORDER'

	#成行注文
	def market_order(self, kind, instrument, units):
		data = self.arrange_data_for_market(kind, instrument, units)					
		resp = self.order.create_order(data)
		#for debug
		print(resp.text)

		#stateは変化させない
		#self.state = 'WAIT_ORDER'

	def normalize_vec(self, x):
		return (x - x.min()) / (x.max() - x.min())

	def preset_prediction(self, x, load=False):
		print('preset in')
		if load is True:	
			self.predicted = self.rnn.predict(x)
			return True
		else:
			return False

	def switch_state(self):
		if self.state == 'ORDER':
			self.state = 'WAIT_ORDER'

		elif self.state == 'WAIT_ORDER':
			self.state = 'POSITION'

		elif self.state == 'POSITION':
			self.state = 'WAIT_POSITION'

		elif self.state == 'WAIT_POSITION':
			self.state = 'ORDER'
		
		else:
			pass

def main():
	#timeframes = {
	#	'15min': 60,
	#	'4H': 45
	#}
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
