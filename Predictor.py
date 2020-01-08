import numpy as np

from RNNparam import RNNparam
from predict_RNN import RNNPredictor

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

		#self.x_max = None
		#self.x_min = None
		#self.is_normalized = False

	def reset(self):
		self.predicted = None
		self.loaded = False

	"""
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
	"""

	def to_rnn_input(self, x, elem='close'):
		#if have single element
		if elem == 'open' or elem == 'high' or elem == 'low' or elem == 'close':
			return np.array([[x]]).transpose(0,2,1)
		else:
			#elem='ohlc'
			#for OHLC data
			pass

	def is_fall(self, candlestick):
		if self.is_rise(candlestick) is not True:
			return True
		else:
			return False
		"""
		#predicted = self.predicted if loaded is True else self.rnn.predict(candlestick)
		if self.loaded is True:
			predicted = self.predicted
		else:
			x_rnn = self.to_rnn_input(candlestick)
			predicted = self.rnn.predict(x_rnn)

		if np.sign(predicted - candlestick[-1]) < 0:
			return True
		else:
			return False
		"""

	def is_range(self, candlestick):
		if self.is_trend(candlestick) is not True:
			return True
		else:
			return False
		"""
		#predicted = self.predicted if loaded is True else self.rnn.predict(candlestick)

		if self.loaded is True:
			predicted = self.predicted
		else:
			x_rnn = self.to_rnn_input(candlestick)
			predicted = self.rnn.predict(x_rnn)

		threshold = 0.2
		diff = np.abs(predicted - candlestick[-1])
		print('range {}'.format(diff[0,0]))
		if diff[0,0] < threshold:
			print('is_range True')
			return True
		else:
			print('is_range false')
			return False
		"""

	def calc_predict(self, candlestick):
		if self.is_loaded() is True:
			return self.predicted
		else:
			return self.rnn.predict(self.to_rnn_input(candlestick))

	def is_rise(self, candlestick):
		#predicted = self.predicted if loaded is True else self.rnn.predict(candlestick)
		"""
		if self.loaded is True:
			predicted = self.predicted
		else:
			x_rnn = self.to_rnn_input(candlestick)
			predicted = self.rnn.predict(x_rnn)
		"""
		predicted = self.calc_predict(candlestick)

		if np.sign(predicted - candlestick[-1]) > 0:
			return True
		else:
			return False

	def is_trend(self, candlestick):
		#predicted = self.predicted if loaded is True else self.rnn.predict(candlestick)
		"""
		if self.loaded is True:
			predicted = self.predicted
		else:
			x_rnn = self.to_rnn_input(candlestick)
			predicted = self.rnn.predict(x_rnn)
		"""

		predicted = self.calc_predict(candlestick)

		print(predicted)
		print(candlestick[-1])

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
		x_rnn = self.to_rnn_input(x)
		self.predicted = self.rnn.predict(x_rnn)
		self.loaded = True
		print('preset prediction')
		return self.predicted

	def is_loaded(self):
		if self.loaded is True:
			return True
		else:
			return False
