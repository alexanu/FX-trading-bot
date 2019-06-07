class CommonParam:
	def __init__(self):
		self.timeframes = {
			'15min': 45,
			'4H': 45
		}
		self.timelist = list(self.timeframes.keys())
		self.entry_freq = '15min'
		self.target = '4H'