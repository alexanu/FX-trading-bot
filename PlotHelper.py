from Plotter import Plotter
from Notify import notify_from_line
from CommonParam import CommonParam

class PlotHelper:
	def __init__(self, timeframes=None):
		if timeframes is None:
			common_param = CommonParam()
			self.timeframes = common_param.timelist
		else:
			self.timeframes = list(timeframes)
		self.nrows = len(self.timeframes)
		self.ncols = 1
		self.figsize = (9, 3 * self.nrows)

	def begin_plotter(self):
		self.plotter = Plotter(nrows=self.nrows, ncols=self.ncols, figsize=self.figsize)

	def end_plotter(self, file_name, notify):
		for i, k in enumerate(self.timeframes):
			self.plotter.set_title(k, i)
		self.plotter.savefig(file_name)
		self.plotter.close()
		if notify is True:
			notify_from_line(f'Output {file_name}', image=file_name)

	def add_tail_oc_slope(self, candlesticks, slopes, intercepts, num_sets):
		for i, k in enumerate(self.timeframes):
			if k in slopes:
				self.plotter.plot_tail_oc_slope(candlesticks[k].ohlc,slopes[k],intercepts[k],num_sets[k], i)

	def add_zebratail(self, candlesticks, num_sets):
		for i, k in enumerate(self.timeframes):
			if k in num_sets:
				self.plotter.plot_ichimatsu(candlesticks[k].ohlc, num_sets[k], i)

	def add_candlestick(self, candlesticks):
		for i, k in enumerate(self.timeframes):
			if k in candlesticks:
				self.plotter.plot_ohlc(candlesticks[k].ohlc, i)

	def add_trendline(self, candlesticks, slopes, intercepts):
		for i, k in enumerate(self.timeframes):
			if k in slopes:
				self.plotter.plot_trendline(candlesticks[k].ohlc,slopes[k],intercepts[k], i)

	def add_double_trendline(self, candlesticks, slopes, intercepts):
		for i, k in enumerate(self.timeframes):
			if k in slopes['high']:
				self.plotter.plot_trendline(candlesticks[k].ohlc,slopes['high'][k],intercepts['high'][k], i)
			if k in slopes['low']:
				self.plotter.plot_trendline(candlesticks[k].ohlc,slopes['low'][k],intercepts['low'][k], i)

	def add_predicted(self, x, predicted):
		for i, k in enumerate(self.timeframes):
			if k in self.predicted:
				self.plotter.plot_predict(x[k], predicted[k], i)

	def add_line(self, x):
		for i, k in enumerate(self.timeframes):
			if k in x_ordered:
				self.plotter.plot(x[k], i)
