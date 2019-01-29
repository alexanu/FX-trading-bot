import numpy as np
import matplotlib.pyplot as plt

class Logger:
	def __init__(self, path):
		self.path = path
		self.figsize = (9, 6)
		self.count = 0
		self._log = {'win_rate': []}

	def plot_loss(self, log):
		fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9,6), squeeze=False)
		axes[0,0].plot(log['loss'])
		plt.savefig(f'{self.path}/loss.png')
		plt.close()

	def plot_predict(self, predict, correct, x_batch, epoch, with_correct=True):
		error_th = 0.0001
		correct = np.append(x_batch, correct)
		predict = np.append(x_batch, predict)
		top = np.max(predict) if np.max(predict) > np.max(correct) else np.max(correct)
		btm = np.min(predict) if np.min(predict) < np.min(correct) else np.min(correct)
		print(top-btm)

		fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9,6), squeeze=False)
		axes[0,0].set_ylim(btm,top)
		if with_correct is True:
			axes[0,0].plot(correct, color='red', linestyle='dashed')
		axes[0,0].plot(predict, linestyle='solid')
		suffix = '_spec' if np.abs(correct[-1] - predict[-1]) < error_th else ''
		plt.savefig(f'{self.path}{suffix}/{epoch}.png')
		plt.close()

	def accumulate_winrate(self, predict, correct, x_batch, epoch):
		epoch_th = 300
		predict = predict[0,0] - x_batch[0,-1,0]
		correct = correct[0,0] - x_batch[0,-1,0]
		if epoch > epoch_th:
			self.count += 1 if np.sign(predict) == np.sign(correct) else 0
			win_rate = self.count / (epoch - epoch_th)
			self._log['win_rate'].append(win_rate)
			print(f'win rate: {win_rate}')

	def diff(self, predict, correct):
		error_th = 0.1
		msg = 'yay!!' if np.abs(predict - correct) < error_th else 'error'
		print(f'{msg}: {predict[0,0] - correct[0,0]}')
