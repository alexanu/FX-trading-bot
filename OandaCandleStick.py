import pandas as pd
import numpy as np

"""
<Alias>    <Description>
W          weekly frequency
M          month end frequency
H          hourly frequency
T,min      minutely frequency
S          secondly frequency

Refference from pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
"""

class CandleStick:
	def __init__(self, rate, count=None):
		self.tickdata = pd.Series([])
		self.ohlc = None
		self.rate = rate
		self.count = count
		self.removable = 1#default
#		self.recv = None

	def set_removable_stick(self, rate):
		t_index = pd.date_range("2017-1-1 00:00", periods=2, freq=rate)
		ts = pd.Series(np.random.randn(len(t_index)), index=t_index)
		self.removable = len(ts.asfreq(self.rate)) - 1

	def append_tickdata(self, recv):
		tick =  pd.Series(float(recv["bids"][0]["price"]),index=[pd.to_datetime(recv["time"])])
		self.tickdata = self.tickdata.append(tick)

	def to_ohlc(self, rate):
		#print(self.tickdata.resample(rate).ohlc())
		return self.tickdata.resample(rate).ohlc()

	def clear_tickdata(self):
		self.tickdata = pd.Series([])

	def concat_ohlc(self, df):
		self.ohlc = pd.concat([self.ohlc, df])

	def concat_ohlc_(self, df):
		print(f'concat1: {self.ohlc.index[-1]}')
		print(f'concat2: {df.index}')
		tmp = pd.concat([self.ohlc[self.ohlc.index.isin(df.index)], df[df.index.isin(self.ohlc.index)]])
		#print(tmp)

		grouped = tmp.groupby(level=0)
		#print(grouped.max()['high'])
		#print(grouped.min()['low'])
		#print(self.ohlc[self.ohlc.index.isin(df.index)]['open'])

		merged = pd.DataFrame({
			'open': self.ohlc[self.ohlc.index.isin(df.index)]['open'],
			'high': grouped.max()['high'],
			'low': grouped.min()['low'],
			'close': df[df.index.isin(self.ohlc.index)]['close']
		})
		merged = merged.loc[:, ['open', 'high', 'low', 'close']]
		#print(merged)

		self.ohlc[self.ohlc.index.isin(df.index)] = merged
		self.ohlc = pd.concat([self.ohlc, df[~df.index.isin(self.ohlc.index)]])
			
	def drop_ohlc(self):
		tmp_index = np.array([])
		for i in range(self.removable):
			tmp_index = np.append(tmp_index,self.ohlc.index[i])
		self.ohlc = self.ohlc.drop(tmp_index)

	def drop_ohlc_(self):
		tmp_index = np.array([])
		print(np.abs(self.count - len(self.ohlc)))
		for i in range(np.abs(self.count - len(self.ohlc))):
			tmp_index = np.append(tmp_index,self.ohlc.index[i])
		self.ohlc = self.ohlc.drop(tmp_index)


	def update_ohlc(self):
		self.concat_ohlc(self.to_ohlc(self.rate))
		self.drop_ohlc()
		self.clear_tickdata()

	def update_ohlc_(self):
		self.concat_ohlc_(self.to_ohlc(self.rate))
		self.drop_ohlc_()
		self.clear_tickdata()

#	def can_update_ohlc(self, rate, size=0):
#		dummy_tick = pd.Series(float(self.recv["tick"]["bid"]),index=[pd.to_datetime(self.recv["tick"]["time"])])
#		dummy_tickdata = self.tickdata.append(dummy_tick)
#		dummy_ohlc = dummy_tickdata.resample(rate).ohlc()
#		if((size + 2) <= len(dummy_ohlc)):
#			return True
#		else:
#			return False

	def can_update(self, recv, mode=None):
		dummy_tick = pd.Series(float(recv['bids'][0]['price']),index=[pd.to_datetime(recv['time'])])
		dummy_tickdata = self.tickdata.append(dummy_tick)
		dummy_ohlc = dummy_tickdata.resample(self.rate).ohlc()

		num_candle = candlestick.count if mode is 'init' else 0

		if((num_candle + 2) <= len(dummy_ohlc)):
			return True
		else:
			return False

	def normalize(self):
		#numpy
		return (self.ohlc - self.ohlc.values.min()) / (self.ohlc.values.max() - self.ohlc.values.min())

	def normalize_each(self):
		#pandas
		return (self.ohlc - self.ohlc.min()) / (self.ohlc.max() - self.ohlc.min())


	"""
		list = ['open', 'high', 'low', 'close']
		normalised_vec = [normalize_by(key) for key in list]
	"""
	def normalize_by(self, key='close', raw=False):
		if raw is True:
			return ((self.ohlc[key] - self.ohlc[key].values.min()) / (self.ohlc[key].values.max() - self.ohlc[key].values.min())).values
		else:
			return (self.ohlc[key] - self.ohlc[key].values.min()) / (self.ohlc[key].values.max() - self.ohlc[key].values.min())

def main():
	print("This class is FXData")


if __name__ == "__main__":
	main()
