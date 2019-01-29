import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import compress

#User-defined module
from RNN import RNN
from dataset_kebin.candle import load_candle
#from dataset_sin.toy_problem import load_seq_sin
#from Layer import Layer
#from Loss import Loss

def main():
	#Layer config
	input_dim = 1
	output_dim = 1
	hidden_dims = [2] #hidden_dims is stored in list for future expansion
	actvs = ['tanh'] # actvs may store activation function for each cell 

	#Cell config -> RNN Class member variable
	tau = 12
	hidden_units = hidden_dims[0]
	keep_prob = 1.0

	model = RNNEvaluator()
	model.set_config(input_dim, output_dim, hidden_dims, actvs, tau, keep_prob)

	if model.should_restore() is True:
		print('restore model')
	else:
		print('brandnew model')

	candle_info = {
		'preproc': 'norm',
		'x_form': None,
		't_form': ['average', 12], #t_form: ['shift', 0]
		'rate' : '5min',
		'price' : 'close',
		'tau' : tau,
	}
	(x_train, t_train), (x_test, t_test) = load_candle(candle_info)

	train_size = int(len(x_train) * 0.8)
	valid_size = len(x_train) - train_size

	#Divide train data for validation
	x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train, test_size=valid_size)

	accuracy = model.evaluate(x_test, t_test)
	print(accuracy)

class RNNEvaluator(RNN):
	def __init__(self, load=True, save=False):
		super().__init__(load, save)

	def evaluate(self, x_test, t_test, y=None):
		print(f'x_test: {x_test.shape}')
		print(f't_test: {t_test.shape}')

		x = tf.placeholder(tf.float32, shape=self.shapes['x'], name='x')
		t = tf.placeholder(tf.float32, shape=self.shapes['t'], name='t')
		batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')

		#graph = tf.get_default_graph()
		#x = graph.get_tensor_by_name('x:0')
		#t = graph.get_tensor_by_name('t:0')
		#batch_size = graph.get_tensor_by_name('batch_size:0')

		y = self.infer(x, batch_size)
		loss_op = self.calc_loss(y, t)
		train_op = self.train(loss_op)

		ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
		if self.should_restore() is True:
			print(ckpt.model_checkpoint_path)
			self.restore_session(ckpt.model_checkpoint_path)
		else:
			print('you have to use saved model')
			return None

		feed_dict = {x : x_test, batch_size : x_test.shape[0]}
		_ = y.eval(session=self._sess, feed_dict=feed_dict)

		feed_dict = {x : x_test, t : t_test, batch_size : x_test.shape[0]}
		accuracy = loss_op.eval(session=self._sess, feed_dict=feed_dict)

		return accuracy

def debug_print(a):
	print('<for debug> : {}'.format(a))

if __name__ == '__main__':
	main()
