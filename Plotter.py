import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mpl_finance as mpf

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
