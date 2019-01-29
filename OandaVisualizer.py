import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.finance as mpf

class Visualizer:
	def __init__(self):
		self.fig = None
		self.axes = None

	def output_candlestick(self, ohlc, rate):
		self.fig, self.axes = plt.subplots(ncols=1, nrows=1, squeeze=False)
		time = ohlc.index[-1].strftime('%Y-%m-%d %H:%M:%S')
		iohlc = np.vstack((range(len(ohlc)), ohlc.values.T)).T
		mpf.candlestick_ohlc(self.axes[0,0], iohlc, width=0.7, colorup='g', colordown='r')
		self.axes[0,0].grid()
		self.axes[0,0].set_ylim(np.min(ohlc["low"]), np.max(ohlc["high"]))
		output_path = './' + rate + '/' + time + '.png'
		self.fig.savefig(output_path)
