import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

#User-defined module
from RNN import RNN
from RNNparam import RNNparam
from dataset_kebin.candle import load_candle
#from dataset_sin.toy_problem import load_seq_sin
#from Layer import Layer
#from Loss import Loss

def main():
	rate = '5min'
	print(rate)
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

	#if key == '4H': 
		#Cell config -> RNN Class member variable
	#	tau = 60
	#	t_form = ['average', 15]


	#if key == '15min':
	#	tau = 45
	#	t_form = ['average', 15]

	candle_param = {
		'preproc': param.trainer_candle['preproc'],
		'x_form': param.trainer_candle['x_form'],
		't_form': param.trainer_candle['t_form'], #t_form: ['shift', 0]
		'rate' : param.trainer_candle['rate'],
		'price' : param.trainer_candle['price'],
		'tau' : param.trainer_candle['tau'],
	}

	model = RNNTrainer(rate=rate, load=False, save=True)
	model.set_config(input_dim, output_dim, hidden_dims, actvs, tau, keep_prob)

	if model.should_restore() is True:
		print('restore model')
	else:
		print('brandnew model')

	(x_train, t_train), (x_test, t_test) = load_candle(candle_param)

	train_size = int(len(x_train) * 0.8)
	valid_size = len(x_train) - train_size

	#Divide train data for validation
	x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train, test_size=valid_size)

	print('train')
	epochs = 1000
	batch_size = 128

	trained_y = model.fit(x_train, t_train, x_valid, t_valid, epochs, batch_size)
	accuracy = model.evaluate(x_test, t_test, y=trained_y)
	print(f'evaluate(loss): {accuracy}')

class RNNTrainer(RNN):
	def evaluate(self, x_test, t_test, y=None):
		print(f'x_test: {x_test.shape}')
		print(f't_test: {t_test.shape}')

		#x = tf.placeholder(tf.float32, shape=self.shapes['x'], name='x')
		#t = tf.placeholder(tf.float32, shape=self.shapes['t'], name='t')
		#batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')
		with self._graph.as_default():
			x = self._graph.get_tensor_by_name(f'x:0')
			t = self._graph.get_tensor_by_name(f't:0')
			batch_size = self._graph.get_tensor_by_name(f'batch_size:0')

			#acc_op = self.calc_loss(y, t)

		feed_dict = {x : x_test, batch_size : x_test.shape[0]}
		_ = y.eval(session=self._sess, feed_dict=feed_dict)

		feed_dict = {x : x_test, t : t_test, batch_size : x_test.shape[0]}
		accuracy = self.acc_op.eval(session=self._sess, feed_dict=feed_dict)

		return accuracy

def debug_print(a):
	print('<for debug> : {}'.format(a))

if __name__ == '__main__':
	main()
