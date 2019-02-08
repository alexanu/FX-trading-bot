import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import compress

#User-defined module
#from Layer import Layer
#from Loss import Loss
from Logger import Logger
from EarlyStopping import EarlyStopping

class RNN:
	def __init__(self, rate='4H', load=False, save=False):
		self.rate = rate

		self.tau = None
		self.shapes = None
		self.cell_config = None

		self._W = None
		self._b = None

		self._log = {"loss": []}

		self._sess = None
		self._graph = {}
		self.ckpt_dir = f'./Save/{rate}'
		self.ckpt_path = f'{self.ckpt_dir}/model.ckpt'

		self.is_load = load
		self.is_save = save


	def __del__(self):
		self._sess.close()

	def batch_normalize(self, x, t):
		norm = np.max(x)
		return x / norm , t / norm

	def calc_accuracy(self, y, t):
		#same as calc_loss
		return tf.reduce_mean(tf.square(y - t))

	def _calc_loss(self, y, t, loss_func='mse'):
		#call Loss class and set loss function
		loss = Loss(loss_func).switch_loss()
		return loss.calc(y, t)

	def calc_loss(self, y, t, loss_func='mse'):
		return tf.reduce_mean(tf.square(y - t))

	def delete_session(self):
		self._sess.close()

	def define_graph(self):
		self._graph = tf.Graph()
		with self._graph.as_default():
			#define variable for TensorFlow
			x = tf.placeholder(tf.float32, shape=self.shapes['x'], name=f'x')
			t = tf.placeholder(tf.float32, shape=self.shapes['t'], name=f't')
			batch_size = tf.placeholder(tf.int32, shape=[], name=f'batch_size')
			keep_prob = tf.placeholder(tf.float32, name=f'keep_prob')
			print('define placeholder')

			#define RNN model
			y = self.infer(x, batch_size)
			loss_op = self.calc_loss(y, t)
			train_op = self.train(loss_op)
			
			#評価用
			self.acc_op = self.calc_accuracy(y, t)
			print('define model')

			self._sess = tf.Session(graph=self._graph)

			ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
			if self.should_restore() is True:
				print(ckpt.model_checkpoint_path)

				#self.restore_session(ckpt.model_checkpoint_path)
				saver = tf.train.Saver()
				saver.restore(self._sess, ckpt.model_checkpoint_path)
			else:
				#initialize tf.Variable
				init = tf.global_variables_initializer()
				self._sess.run(init)

			return y, loss_op, train_op


	def fit(self, x_train, t_train, x_valid, t_valid, epochs, n_batch):

		print(f'x_train: {x_train.shape}')
		print(f't_train: {t_train.shape}')
		print(f'x_valid: {x_valid.shape}')
		print(f't_valid: {t_valid.shape}')
		#pevaluator = PermanentEvaluator()
		#ievaluator = InstantEvaluator()
		#visualizer = NNVisualizer('./Output')
		logger = Logger('./Output')

		earlystopping = EarlyStopping(patience=50, verbose=False)

		"""
		self._graph = tf.Graph()
		with self._graph.as_default():
			#define variable for TensorFlow
			x = tf.placeholder(tf.float32, shape=self.shapes['x'], name=f'x_{self.rate}')
			t = tf.placeholder(tf.float32, shape=self.shapes['t'], name=f't_{self.rate}')
			batch_size = tf.placeholder(tf.int32, shape=[], name=f'batch_size_{self.rate}')
			keep_prob = tf.placeholder(tf.float32, name=f'keep_prob_{self.rate}')
			print('define placeholder')

			#define RNN model
			y = self.infer(x, batch_size)
			loss_op = self.calc_loss(y, t)
			train_op = self.train(loss_op)

			self.acc_op = self.calc_accuracy(y, t)
			print('define model')

			self._sess = tf.Session(graph=self._graph)

			ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
			if self.should_restore() is True:
				print(ckpt.model_checkpoint_path)

				#self.restore_session(ckpt.model_checkpoint_path)
				saver = tf.train.Saver()
				saver.restore(self._sess, ckpt.model_checkpoint_path)
			else:
				#initialize tf.Variable
				init = tf.global_variables_initializer()
				self._sess.run(init)
		"""
		y, loss_op, train_op = self.define_graph()
		with self._graph.as_default():
			x = self._graph.get_tensor_by_name(f'x:0')
			t = self._graph.get_tensor_by_name(f't:0')
			batch_size = self._graph.get_tensor_by_name(f'batch_size:0')

			var_name_list = [v.name for v in tf.trainable_variables()]
			print(var_name_list)

			batchs = x_train.shape[0] // n_batch

			#print('start inference')
			for epoch in range(epochs):
				#乱数が重複しないように変更する
				for i in range(batchs):
					###
					#Train Operation
					###
					batch_mask = np.random.choice(x_train.shape[0], n_batch)
					x_batch = x_train[batch_mask]
					t_batch = t_train[batch_mask]

					#x_batch, t_batch = self.batch_normalize(x_batch, t_batch)
					feed_dict = {x: x_batch, t: t_batch, batch_size: n_batch}
					train_op.run(session=self._sess, feed_dict=feed_dict)
					
				###
				#Calculate Loss
				###
				#x_valid, t_valid = self.batch_normalize(x_valid, t_valid)
				feed_dict = {x: x_valid, t: t_valid, batch_size: x_valid.shape[0]}
				loss = loss_op.eval(session=self._sess, feed_dict=feed_dict)
				self._log["loss"].append(loss)
				print(f'epoch: {epoch}, loss: {loss}')

				###
				#Try Prediction
				###
				#pick one sequential data (to give batch_size = 1)
				batch_mask = np.random.choice(x_valid.shape[0], 1)
				x_batch = x_valid[batch_mask]
				t_batch = t_valid[batch_mask]

				#x_batch, t_batch = self.batch_normalize(x_batch, t_batch)
				feed_dict = {x: x_batch, batch_size: 1}
				predict = y.eval(session=self._sess, feed_dict=feed_dict)

				#
				#
				#
				logger.diff(predict, t_batch)
				logger.accumulate_winrate(predict, t_batch, x_batch, epoch)
				logger.plot_predict(predict, t_batch, x_batch, epoch)
				print('')
				
				if earlystopping.validate(loss) is True:
					break
			#
			#
			#
			logger.plot_loss(self._log)
			if self.is_save is True:
				#self.save_session()
				saver = tf.train.Saver()
				saver.save(self._sess, self.ckpt_path)

		self._graph.finalize()
		return y

	def finalize_graph(self):
		self._graph.finalize()

	def init_uninitialized_variable(self):
		not_initialized_vars = self.get_uninitialized_variables()
		print(len(not_initialized_vars))
		self._sess.run(tf.variables_initializer(not_initialized_vars))

	def get_uninitialized_variables(self):
		#変数(tf.Variable)をすべて取得
		g_vars = tf.global_variables()
		#初期化されていない変数をbool型のlistで取得
		is_not_initialized = self._sess.run([~(tf.is_variable_initialized(v)) for v in g_vars])
		#初期化されていないtf.Variableをlistに格納
		not_initialized_vars = list(compress(g_vars, is_not_initialized))

		return not_initialized_vars

	def get_initialized_variables(self):
		#変数(tf.Variable)をすべて取得
		g_vars = tf.global_variables()
		#初期化されている変数をbool型のlistで取得
		is_initialized = self._sess.run([~(tf.is_variable_initialized(v)) for v in g_vars])
		#初期化されているtf.Variableをlistに格納
		initialized_vars = list(compress(global_vars, is_initialized))

		return initialized_vars

	def has_ckpt(self):
		ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
		if ckpt:
			return True
		else:
			return False

	def infer(self, x, batch_size):
		#dropout has not been implemented yet
		#cell = tf.contrib.rnn.BasicRNNCell(hidden_units)# hidden_units = self.hidden_dims[0]
		cell = tf.contrib.rnn.BasicLSTMCell(self.cell_config['n_units'], activation=tf.tanh, name=f'basic_lstm_cell_{self.rate}')
		initial_state = cell.zero_state(batch_size, tf.float32)

		state = initial_state
		cell_outputs = []
		with tf.variable_scope("RNN"):
			for t in range(self.tau):
				if t > 0:
					tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(x[:,t,:], state)
				cell_outputs.append(cell_output)
		output = cell_outputs[-1]

		
		#y = Wx + b(W: weight, b: bias)
		self._W = self.init_weight(self.shapes['W'])
		self._b = self.init_bias(self.shapes['b'])
		y = tf.matmul(output, self._W) + self._b
		return y
	
	def init_bias(self, shape):
		initial = tf.zeros(shape)
		return tf.get_variable(f'bias_{self.rate}', initializer=initial)
		#return tf.Variable(initial)

	def init_weight(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.01)
		return tf.get_variable(f'weight_{self.rate}', initializer=initial)
		#return tf.Variable(initial)

	def restore_session(self, ckpt_path):
		saver = tf.train.Saver()
		saver.restore(self._sess, ckpt_path)

	def save_session(self):
		saver = tf.train.Saver()
		saver.save(self._sess, self.ckpt_path)

	def set_config(self, input_dim, output_dim, hidden_dims, actvs, tau, keep_prob):
		self.tau = tau
		hidden_units = hidden_dims[0]

		x_shape = [None, tau, input_dim]
		t_shape = [None, output_dim]
		W_shape = [hidden_units, output_dim]
		b_shape = [output_dim]

		self.shapes = {
			'x': x_shape,
			't': t_shape,
			'W': W_shape,
			'b': b_shape
		}

		self.cell_config = {'n_units': hidden_units}
		

	def set_dropout(self, src, keep_prob):
		return tf.nn.dropout(src, keep_prob)

	def should_restore(self):
		if self.is_load is True and self.has_ckpt() is True:
			return True
		else:
			return False

	def train(self, loss):
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
		train_step = optimizer.minimize(loss)
		return train_step

	"""
	def evaluate(self, x_test, t_test, y=None):
		if y is None:
			y = self.infer(self._x, self._keep_prob)
			loss_op = self.calc_loss(y, self._t)
			train_op = self.train(loss_op)

			#acc_op = self.calc_accuracy(y, self._t)
			acc_op = self.calc_loss(y, self._t)

			ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
			if self.should_restore() is True:
				print(ckpt.model_checkpoint_path)
				self.restore_session(ckpt.model_checkpoint_path)
			else:
				init = tf.global_variables_initializer()
				self._sess.run(init)

			_ = y.eval(session=self._sess,
				feed_dict = 
				{
					self._x : x_test,
					self.batch_size : x_test.shape[0]
				}
			)

		else:
			#acc_op = self.calc_accuracy(self._y, self._t)
			#acc_op = self.calc_loss(self._y_in_fit, self._t)
			acc_op = self.calc_loss(y, self._t)

			_ = y.eval(session=self._sess,
				feed_dict = 
				{
					self._x : x_test,
					self.batch_size : x_test.shape[0]
				}
			)

		accuracy = acc_op.eval(session=self._sess,
			feed_dict = 
			{
				self._x : x_test,
				self._t : t_test,
				self.batch_size : x_test.shape[0]
			}
		)
		return accuracy
	"""

def debug_print(a):
	print('<for debug> : {}'.format(a))

if __name__ == '__main__':
	main()
