import numpy as np
import pandas as pd
import requests
import time

#User defined classes
from OandaEndpoints import Pricing
from OandaCandleStick import CandleStick
from Trader import Trader
from Evaluator import Evaluator
from PlotHelper import PlotHelper
from Notify import notify_from_line
from OandaEndpoints import from_byte_to_dict, from_response_to_dict
from Analysis import is_rise_with_trendline

class Manager:

	"""

	売買のタイミングを管理するクラス
	"""
	
	def __init__(self, param: list, instrument: str, environment: str='demo', mode: str='test'):
		"""

		トレードの対象となる通貨ペアと時間足（種類と本数）を定義

		Parameter
		---------
		param : list
			取り扱う時間足のリスト
		instrument : str
			取り扱う通貨ペア
		environment : str
			トレード環境（本番or仮想)を指定
		mode : str
			デバッグ用の出力を行うかどうかを指定

		Self
		----
		param : list
			取り扱う時間足のリスト
		instrument : str
			取り扱う通貨ペア
		environment : str
			トレード環境（本番or仮想)を指定
		trader : Trader (User defined)
		evaluator : Evaluator (User defined)
		checking_freq : int (const)
			画面出力の頻度を決定するための定数
		count : int
			画面出力のための内部カウンタ
		

		"""

		self.instrument = instrument
		self.environment = environment
		self.param = param
		self.trader = Trader(instrument, environment, mode)
		#self.predictors = {k: Predictor(k) for k in param.timelist}
		#self.fetters = {k: Fetter(k) for k in param.timelist}
		self.evaluator = Evaluator(self.param.timelist, instrument, environment)

		self.checking_freq = 10
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

		#現在保有しているポジションをすべて決済
		self.trader.clean()
		print('<Manager> Close all positions')

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

		戦略を元に売買を実行する

		Parameter
		---------
		resp : requests.response
			ストリーミングでの接続を行うためのレスポンスデータ
		candlesticks : dict
			取り扱うローソク足の種類と本数を格納した辞書型

		Exception
		---------
		ValueError
		urllib3.exceptions.ProtocolError
		requests.exceptions.Chunked EncodingError

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

	def execute_strategy(self, recv: dict, candlesticks: dict):
		
		"""
		売買の戦略を決定し、売買タイミングを決定する

		Parameter
		---------
		recv : dict
			為替のtickデータが格納された辞書型のデータ
		candlesticks : dict
			1つ、またはそれ以上のローソク足の組

		"""

		for k, v in candlesticks.items():
			if v.can_update(recv) is True:
				v.update_ohlc_()
				print(f'{k} is updated -> total length : {len(v.ohlc)}')

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

	def entry(self, candlesticks: dict) -> bool:

		"""

		エントリー状態に遷移させる
		is_rise_with_xxx（任意のアルゴリズム）によって戦略を決定
		現状、「買い」（今後上昇するかのみ）のための設計
		
		Parameter
		---------
		candlesticks : dict
			1つ、またはそれ以上のローソク足の組

		Exception
		---------
		各is_rise_with_xxxが投げる例外を参照、基本的にValueErrorを推奨

		"""

		if self.trader.state == 'ORDER':
			is_rises = []
			error_count = 0

			"""

			Template(now)
			-------------
			try:
				is_rises.append(is_rise_with_xxx(arg1, arg2, ...))
			except ValueError as e:
				print(f'{e}')
				error_count += 1

			try:
				is_rises.append(is_rise_with_yyy(arg1, arg2, ...))
			except ValueError as e:
				print(f'{e}')
				error_count += 1

			> Template(in future)
			> -------------------

			> try:
			> 	is_rises.append(is_rise_with_xxx(arg1, arg2, ...))
			> 	is_rises.append(is_rise_with_yyy(arg1, arg2, ...))
			> except ValueError as e:
			> 	print(f'{e}')
			> 	error_flag = True


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

