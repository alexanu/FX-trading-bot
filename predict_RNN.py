import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import compress

#User-defined module
from RNN import RNN
from RNNparam import RNNparam
from dataset_kebin.candle import load_candle
#from dataset_sin.toy_problem import load_seq_sin
#from Layer import Layer
#from Loss import Loss

def main():
	rates = ['1min', '5min']

	for rate in rates:
		param = RNNparam(rate)

		#Layer config
		input_dim = param.input_dim
		output_dim = param.output_dim
		hidden_dims = param.hidden_dims #hidden_dims is stored in list for future expansion
		actvs = param.actvs # actvs may store activation function for each cell

		#Cell config -> RNN Class member variable
		tau = param.tau
		hidden_units = param.hidden_units
		keep_prob = param.keep_prob

		candle_info = {
			'preproc': param.trainer_candle['preproc'],
			'x_form': param.trainer_candle['x_form'],
			't_form': param.trainer_candle['t_form'], #t_form: ['shift', 0]
			'rate' : param.trainer_candle['rate'],
			'price' : param.trainer_candle['price'],
			'tau' : param.trainer_candle['tau'],
		}
		#if rate == '15min':
			#Cell config -> RNN Class member variable
		#	tau = 60
		#	hidden_units = hidden_dims[0]
		#	keep_prob = 1.0

		#	candle_info = {
		#		'preproc': 'norm',
		#		'x_form': None,
		#		't_form': ['average', 30], #t_form: ['shift', 0]
		#		'rate' : '15min',
		#		'price' : 'close',
		#		'tau' : tau,
		#	}

		#if rate == '4H':
		#	#Cell config -> RNN Class member variable
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

		model = RNNPredictor(rate=rate)
		model.set_config(input_dim, output_dim, hidden_dims, actvs, tau, keep_prob)

		if model.should_restore() is True:
			print('restore model')
		else:
			print('brandnew model')

		(x_train, t_train), (x_test, t_test) = load_candle(candle_info)

		train_size = int(len(x_train) * 0.8)
		valid_size = len(x_train) - train_size

		#Divide train data for validation
		x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train, test_size=valid_size)

		print('preload begin')
		model.preload_model()
		print('preload end')
		x_test = x_test[0,:,:]
		x_test = np.array([x_test])
		print('predict')
		predicted = model.predict(x_test)
		print(predicted)

class RNNPredictor(RNN):
	def __init__(self, rate, load=True, save=False):
		super().__init__(rate, load, save)
		self._y = None

	def predict(self, x_):
		#self.preload_model()
		with self._graph.as_default():
			#graph = tf.get_default_graph()
			#x = self._graph.get_tensor_by_name(f'x:0')
			#batch_size = self._graph.get_tensor_by_name(f'batch_size:0')
			x = self._graph.get_tensor_by_name(f'x:0')
			batch_size = self._graph.get_tensor_by_name(f'batch_size:0')

			feed_dict = {x : x_, batch_size : 1}
			predicted = self._y.eval(session=self._sess, feed_dict=feed_dict)
			return predicted

	def preload_model(self):
		y, _, _ = self.define_graph()
		"""
		self._graph = tf.Graph()
		with self._graph.as_default():
			x = tf.placeholder(tf.float32, shape=self.shapes['x'], name=f'x_{self.rate}')
			t = tf.placeholder(tf.float32, shape=self.shapes['t'], name=f't_{self.rate}')
			batch_size = tf.placeholder(tf.int32, shape=[], name=f'batch_size_{self.rate}')

			y = self.infer(x, batch_size)
			loss_op = self.calc_loss(y, t)
			train_op = self.train(loss_op)

			self._sess = tf.Session(graph=self._graph)

			ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
			if self.should_restore() is True:
				print(ckpt.model_checkpoint_path)
				#self.restore_session(ckpt.model_checkpoint_path)
				saver = tf.train.Saver()
				saver.restore(self._sess, ckpt.model_checkpoint_path)
			else:
				print('you have to use saved model')
				return None
		"""
		ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
		if self.should_restore() is True:
			print(ckpt.model_checkpoint_path)
		else:
			print('you have to use saved model')
			y = None
		self._y = y
		self._graph.finalize()
	
	def evaluate(self, x_test, t_test, y=None):
		print(f'x_test: {x_test.shape}')
		print(f't_test: {t_test.shape}')

		#define variable for TensorFlow
		x = tf.placeholder(tf.float32, shape=self.shapes['x'])
		t = tf.placeholder(tf.float32, shape=self.shapes['t'])
		batch_size = tf.placeholder(tf.int32, shape=[])
		keep_prob = tf.placeholder(tf.float32)
		print('define placeholder')

		y = self.infer(x, batch_size)
		loss_op = self.calc_loss(y, t)
		train_op = self.train(loss_op)
		print('define model')

		ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
		if self.should_restore() is True:
			print(ckpt.model_checkpoint_path)
			self.restore_session(ckpt.model_checkpoint_path)
		else:
			init = tf.global_variables_initializer()
			self._sess.run(init)

		feed_dict = {x : x_test, batch_size : x_test.shape[0]}
		_ = y.eval(session=self._sess, feed_dict=feed_dict)

		feed_dict = {x : x_test, t : t_test, batch_size : x_test.shape[0]}
		accuracy = loss_op.eval(session=self._sess, feed_dict=feed_dict)

		return accuracy


def debug_print(a):
	print('<for debug> : {}'.format(a))

if __name__ == '__main__':
	main()
