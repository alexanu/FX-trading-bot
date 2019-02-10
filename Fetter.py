class Fetter:
	def __init__(self, rate):
		self.rate = rate

	#Strategy 1 for 4H
	def foo(self, ohlc):
		#OHLCデータについて計算
		x = ohlc
		x = 1

		#計算後のデータを返却
		return x

	def is_foo(self, ohlc):
		if self.rate != '4H':
			return False

		hoge = self.foo(ohlc)
		#fooから帰ってきた値に対してTrue or Falseを判断
		if hoge > 0:
			return True
		else:
			return False

	#Strategy 2 for 4H
	def foofoo(self, ohlc):
		#OHLCデータについて計算
		x = ohlc
		x = 1

		#計算後のデータを返却
		return x

	def is_foofoo(self, ohlc):
		if self.rate != '4H':
			return False

		hoge = self.foofoo(ohlc)
		if hoge > 0:
			return True
		else:
			return False

	#Strategy 1 for 15min
	def bar(self, ohlc):
		#OHLCデータについて計算
		x = ohlc
		x = 1

		#計算後のデータを返却
		return x

	def is_bar(self, ohlc):
		if self.rate != '15min':
			return False

		hoge = self.bar(ohlc)
		if hoge > 0:
			return True
		else:
			return False

	#Strategy 2 for 15min
	def barbar(self, ohlc):
		#OHLCデータについて計算
		x = ohlc
		x = 1

		#計算後のデータを返却
		return x

	def is_barbar(self, ohlc):
		if self.rate != '15min':
			return False

		hoge = self.barbar(ohlc)
		if hoge > 0:
			return True
		else:
			return False

	#Strategy 2 for 15min
	def barbarbar(self, ohlc):
		#OHLCデータについて計算
		x = ohlc
		x = 1

		#計算後のデータを返却
		return x

	def is_barbarbar(self, ohlc):
		if self.rate != '1min':
			return False

		hoge = self.barbar(ohlc)
		if hoge > 0:
			return True
		else:
			return False

	#Strategy 2 for 15min
	def barbarbarbar(self, ohlc):
		#OHLCデータについて計算
		x = ohlc
		x = 1

		#計算後のデータを返却
		return x

	def is_barbarbarbar(self, ohlc):
		if self.rate != '5min':
			return False

		hoge = self.barbar(ohlc)
		if hoge > 0:
			return True
		else:
			return False

	def is_through(self, ohlc, is_rise):

		if self.rate == '4H':
			if self.is_foofoo(ohlc) is False:
				return False
			if self.is_foo(ohlc) is False:
				return False

			return True

		elif self.rate == '15min':
			if self.is_bar(ohlc) is False:
				return False
			if self.is_barbar(ohlc) is False:
				return False

			return True

		elif self.rate == '1min':
			if self.is_barbarbar(ohlc) is False:
				return False

			return True

		elif self.rate == '5min':
			if self.is_barbarbarbar(ohlc) is False:
				return False

			return True
		else:
			return False
