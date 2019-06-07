class CommonParam:
	def __init__(self):
		self.timeframes = {
			'5min': 45,
			'15min': 45
		}
		self.timelist = list(self.timeframes.keys())
		self.entry_freq = '5min'
		self.target = '15min'