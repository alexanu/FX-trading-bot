import numpy as np
import pandas as pd

#for Exception
import requests

#for Sleep
import time

#User difined classes
from OandaEndpoints import Pricing
from OandaCandleStick import CandleStick
#from predict_RNN import RNNPredictor
#from RNNparam import RNNparam
#from Fetter import Fetter
#from Plotter import Plotter
#from Predictor import Predictor
from Trader import Trader
from Evaluator import Evaluator
from PlotHelper import PlotHelper

#User difined functions
from Notify import notify_from_line
from OandaEndpoints import from_byte_to_dict, from_response_to_dict
from Analysis import is_rise_with_zebratail, is_rise_with_trendline

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

class Manager:
	def __init__(self, param, instrument, environment='demo', mode='test'):
		self.instrument = instrument
		self.environment = environment
		self.param = param
		self.trader = Trader(instrument, environment, mode)
		#self.predictors = {k: Predictor(k) for k in param.timelist}
		#self.fetters = {k: Fetter(k) for k in param.timelist}
		self.evaluator = Evaluator(self.param.timelist, instrument, environment)

		self.checking_freq = 0
		self.count = 0

	def __del__(self):
		self.trader.clean()

	def has_price(self, msg):
		if msg:
			msg = from_byte_to_dict(msg)
			if msg["type"] == "PRICE":
				return True
			else:
				return False
		else:
			return False

	def has_heartbeat(self, msg):
		if msg:
			msg = from_byte_to_dict(msg)
			if msg["type"] == "HEARTBEAT":
				return True
			else:
				return False
		else:
			return False

	def driver(self, candlesticks):
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
		print('test_driver begin')
		print('trader was created')

		#現在保有しているポジションをすべて決済
		self.trader.clean()
		print('close all position to use trader.clean()')

		while True:
			#指定した通貨のtickをストリーミングで取得する
			pricing_handler = Pricing(self.environment)
			resp = pricing_handler.connect_to_stream(self.instrument)
			try:
				resp.raise_for_status()
			except ConnectionError as e:
				print(f'connect_to_stream : Catch {e}')
				print(f'connect_to_stream : Retry to Connect')
				time.sleep(1)
				continue

			try:
				self.run(resp, candlesticks)
			except requests.exceptions.ChunkedEncodingError as e:
				print(f'run : Catch {e}')
				print(f'run : Retry to Connect')
				time.sleep(1)
				continue

	def run(self, resp, candlesticks):
		"""
		raise ValueError
		raise urllib3.exceptions.ProtocolError
		raise requests.exceptions.Chunked EncodingError
		"""
		for line in resp.iter_lines(1):
			if self.has_price(line):
				recv = from_byte_to_dict(line)
				self.execute_strategy(recv, candlesticks)

			#一定時間ごとに注文or決済が反映されたかを確認する
			if self.checking_freq == self.count:
				print(f'heart beat(span): {self.count}')
				#注文が反映されたか
				if self.trader.test_is_reflected_order() is True:
					#WAIT_ORDER -> POSITION
					self.trader.switch_state()

				#決済が反映されたか
				if self.trader.test_is_reflected_position() is True:
					#WAIT_POSITION -> ORDER
					self.trader.switch_state()
			self.count = 0 if self.checking_freq == self.count else (self.count + 1)

	def execute_strategy(self, recv, candlesticks):
		for k, v in candlesticks.items():
			#if can_update(recv, v) is True:
			if v.can_update(recv) is True:
				v.update_ohlc_()
				print(k)
				print(len(v.ohlc))

				if k == self.param.entry_freq:
					try:
						#エントリー
						self.entry(candlesticks)
					except (RuntimeError, ValueError) as e:
						print(f'{e}')

					#決済（クローズ）
					self.settle(candlesticks)

				#時間足が更新されたときにも注文が反映されたかを確認する
				#注文が反映されたか
				if self.trader.test_is_reflected_order() is True:
					#WAIT_ORDER -> POSITION
					self.trader.switch_state()

				#決済が反映されたか
				if self.trader.test_is_reflected_position() is True:
					#WAIT_POSITION -> ORDER
					self.trader.switch_state()
			v.append_tickdata(recv)

	def entry(self, candlesticks):
		if self.trader.state == 'ORDER':
			is_rises = []
			error_count = 0

			"""
			try:
				length = 10
				is_rises.append(is_rise_with_trendline(self.param.target, candlesticks, length))
			except TypeError as e:
				print(f'{e}')
				error_count += 1

			try:
				is_rises.append(is_rise_with_zebratail(self.param.target, candlesticks))
			except TypeError as e:
				print(f'{e}')
				error_count += 1
			"""
			try:
				is_rises.append(is_rise_with_insidebar(self.param.target, candlesticks, length))
			except ValueError as e:
				print(f'{e}')
				error_count += 1

			try:
				is_rises.append(is_rise_with_line_touch(self.param.target, candlesticks))
			except ValueError as e:
				print(f'{e}')
				error_count += 1

			try:
				is_rises.append(is_rise_with_trendline_maxarea(self.param.target, candlesticks))
			except ValueError as e:
				print(f'{e}')
				error_count += 1

			if error_count > 0:
				print(f'Error count : {error_count}')
				raise RuntimeError('entry : error count is not 0')

			if (all(is_rises) is True or any(is_rises) is False) is False:
				print('Unstable: ENTRY')
				raise ValueError('entry : is_rises is not [All True] or [All False]')

			is_rise = all(is_rises)
			kind = 'BUY' if True is is_rise else 'SELL'

			is_order_created = self.trader.test_create_order(is_rise)
			self.evaluator.set_order(kind, True)
			print(kind)

			if True is is_order_created:
				#ORDER状態からORDERWAITINGに状態遷移
				self.trader.switch_state()

	def settle(self, candlesticks):
		threshold = 8
		if self.trader.state == 'POSITION':
			#ポジションを決済可能か
			if self.trader.can_close_position(threshold) is True:
				#決済の注文を発行する
				is_position_closed = self.trader.test_close_position()

				x = {}
				correct = {}
				for k, v in candlesticks.items():
					x[k] = v.normalize_by('close').values
					#or x[k] = v.normalize_by('close' raw=True)
					correct[k] = np.mean(x[k][-threshold:])

				self.evaluator.set_close(True)

				#LINE notification
				#timeframes = list(candlesticks.keys())
				plothelper = PlotHelper()
				plothelper.begin_plotter()
				plothelper.add_candlestick(candlesticks)
				plothelper.end_plotter('close.png', True)

				self.evaluator.log_score()

				#決済の注文が発行されたか
				if True is is_position_closed:
					#ORDER -> WAIT_ORDER
					self.trader.switch_state()

			else:
				print('Position can not be closed in this update')
			#決済するかを判断するアルゴリズムを更新
			self.trader.update_whether_closing_position(threshold)

