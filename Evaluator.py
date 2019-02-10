class Evaluator:
	def __init__(self, timeframes, instrument, environment='demo'):
		self.pre_price = None
		self.post_price = None
		self.profit = 0
		self.log = {
			'win_rate': [],
			'profit': [0],
			'win_or_lose': []
		}
		self.instrument = instrument
		self.pricing = Pricing(environment)

		self.kind = None

		#取り扱う時間足のリスト
		#list
		self.timeframes = list(timeframes)

		self.nrows = len(timeframes)
		self.ncols = 1
		self.figsize = (9, 3*self.nrows)

	def set_order(self, kind):
		resp = self.pricing.get_pricing_information(self.instrument)
		msg = from_byte_to_dict(resp.content)
		self.pre_price = float(msg['prices'][0]['bids'][0]['price'])
		self.kind = kind

	def set_close(self):
		resp = self.pricing.get_pricing_information(self.instrument)
		msg = from_byte_to_dict(resp.content)
		self.post_price = float(msg['prices'][0]['bids'][0]['price'])

	def begin_plotter(self):
		self.plotter = Plotter(nrows=self.nrows, ncols=self.ncols, figsize=self.figsize)

	def end_plotter(self, file_name, notify):
		for i, k in enumerate(self.timeframes):
			self.plotter.set_title(k, i)
		self.plotter.savefig(file_name)
		self.plotter.close()
		if notify is True:
			notify_from_line(self.kind, image=file_name)
			
	def add_tail_oc_slope(self, candlesticks, slopes, intercepts, num_sets):
		for i, k in enumerate(self.timeframes):
			if k in slopes:
				self.plotter.plot_tail_oc_slope(candlesticks[k].ohlc,slopes[k],intercepts[k],num_sets[k], i)

	def add_ichimatsu(self, candlesticks, num_sets):
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

	def output_ichimatsu(self, candlesticks, slopes, intercepts, num_sets, notify=True):
		plotter = Plotter(nrows=self.nrows, ncols=self.ncols, figsize=self.figsize)
		file_name = 'ichimatsu.png'
		for i, k in enumerate(self.timeframes):
			if k in candlesticks:
				plotter.plot_ohlc(candlesticks[k].ohlc, i)
			if k in slopes:
				plotter.plot_ichimatsu(candlesticks[k].ohlc,slopes[k],intercepts[k],num_sets[k], i)

			#確実に設定できるものを記述
			plotter.set_title(k, i)
		#その他設定
		plotter.savefig(file_name)
		plotter.close()
		if notify is True:
			notify_from_line(self.kind, image=file_name)

	def output_trendline(self, candlesticks, slopes, intercepts, price='high', notify=True):
		plotter = Plotter(nrows=self.nrows, ncols=self.ncols, figsize=self.figsize)
		file_name = f'{price}_trendline.png'
		for i, k in enumerate(self.timeframes):
			if k in candlesticks:
				plotter.plot_ohlc(candlesticks[k].ohlc, i)
			if k in slopes:
				plotter.plot_trendline(candlesticks[k].ohlc,slopes[k],intercepts[k], i)

			#確実に設定できるものを記述
			plotter.set_title(k, i)
		#その他設定
		plotter.savefig(file_name)
		plotter.close()
		if notify is True:
			notify_from_line(self.kind, image=file_name)

	def output_double_trendline(self, candlesticks, slopes, intercepts, notify=True):
		plotter = Plotter(nrows=self.nrows, ncols=self.ncols, figsize=self.figsize)
		file_name = f'trendline.png'
		for i, k in enumerate(self.timeframes):
			if k in candlesticks:
				plotter.plot_ohlc(candlesticks[k].ohlc, i)
			if k in slopes['high']:
				plotter.plot_trendline(candlesticks[k].ohlc,slopes['high'][k],intercepts['high'][k], i)
			if k in slopes['low']:
				plotter.plot_trendline(candlesticks[k].ohlc,slopes['low'][k],intercepts['low'][k], i)

			#確実に設定できるものを記述
			plotter.set_title(k, i)
		#その他設定
		plotter.savefig(file_name)
		plotter.close()
		if notify is True:
			notify_from_line(self.kind, image=file_name)

	def output_candlestick(self, candlesticks, notify=True):
		plotter = Plotter(nrows=self.nrows, ncols=self.ncols, figsize=self.figsize)
		file_name = 'candlesticks.png'
		for i, k in enumerate(self.timeframes):
			if k in candlesticks:
				plotter.plot_ohlc(candlesticks[k].ohlc, i)

			#確実に設定できるものを記述
			plotter.set_title(k, i)
			#self.axes[i,0].set_title(k, fontsize=18)
		#その他設定
		plotter.set_for_close()
		plotter.savefig(file_name)
		plotter.close()
		if notify is True:
			notify_from_line('close', image=file_name)

	def output_score(self, msg, notify=True):
		plotter = Plotter(nrows=2, ncols=1, figsize=(8,8))
		file_name = 'score.png'
		for i, key in enumerate(['profit', 'win_rate']):
			plotter.plot(self.log[key], i)

			plotter.set_title(key, i)
			#あとでset_for_scoreにつっこむ
			if i == 1:
				plotter.set_ylim(-0.25,1.25, i)
		#その他設定
		plotter.set_for_score()
		plotter.savefig(file_name)
		plotter.close()
		if notify is True:
			notify_from_line(msg, image=file_name)

	def log_score(self):
		diff = self.post_price - self.pre_price
		diff = diff if self.kind == 'BUY' else -diff
		self.log['profit'].append(self.log['profit'][-1] + diff)

		sign = np.sign(self.post_price - self.pre_price)
		sign = sign if self.kind == 'BUY' else -sign
		self.log['win_or_lose'].append(sign)

		msg = self.kind
		msg += ' WIN' if self.log['win_or_lose'][-1] > 0 else ' LOSE'

		print(f'kind: {self.kind} raw_sign: {sign}  diff: {diff}')
		print(f'profit: {self.log["profit"][-1]} {self.log["profit"][-1] + diff}')

		print(msg)	

		win_rate = self.log['win_or_lose'].count(1) / len(self.log['win_or_lose'])
		self.log['win_rate'].append(win_rate)
		print(f'win_rate: {win_rate}')

		self.output_score(msg)
