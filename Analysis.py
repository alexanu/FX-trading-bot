import numpy as np
import pandas as pd
from scipy.stats import linregress

#User difined classes
from PlotHelper import PlotHelper

def is_rise_with_zebratail(src, candlesticks):
	signs, bottom, top = calc_zebratail(src, candlesticks)

	print(f'Top: {top==signs}')
	print(f'Bottom: {bottom==signs}')

	if (bottom != signs) and (top != signs):
		print('Unsteable')
		raise TypeError("is_rise_with_zebra : Return no boolean value(i.e. None)")

	#and np.abs(slopes['5min']) < 0.01
	#底値(BUY)
	if bottom == signs:
		return True
	#高値(SELL)
	else:
		return False

def calc_zebratail(src, candlesticks):
	ohlc = candlesticks[src].ohlc.copy()
	ohlc = (ohlc - ohlc.values.min()) / (ohlc.values.max() - ohlc.values.min())

	#ローソク足を2本1セットとして、numに対となるローソク足の本数を指定
	num_set = 2 * 1
	tail = ohlc[-num_set:]

	close_open = ['close' if i%2 == 0 else 'open' for i in range(num_set)]
	x = np.array([i + 1 for i in range(num_set)])
	y = [tail[co].values[i] for i, co in enumerate(close_open)]

	#meno stderrを使うかもしれない
	slope, intercept, _, _, _ = linregress(x, y)

	bottom = [+1 if i%2 == 0 else -1 for i in range(num_set)]
	top    = [+1 if i%2 != 0 else -1 for i in range(num_set)]
	signs = list(np.sign(tail['open'].values - tail['close'].values))

	output_zebratail(src, candlesticks, slope, intercept, num_set)

	return (signs, bottom, top)

def output_zebratail(src, candlesticks, slope, intercept, num_set):
	slopes = {}
	intercepts = {}
	num_sets = {}

	slopes[src] = slope
	intercepts[src] = intercept
	num_sets[src] = num_set

	#Output
	#timeframes = list(candlesticks.keys())
	plothelper = PlotHelper()
	plothelper.begin_plotter()
	plothelper.add_candlestick(candlesticks)
	plothelper.add_tail_oc_slope(candlesticks, slopes, intercepts, num_sets)
	plothelper.add_zebratail(candlesticks, num_sets)
	plothelper.end_plotter('zebratail.png', True)

def is_rise_with_trendline(src, candlesticks, length):
	h_slopes, h_intercepts = calc_trendline(candlesticks, length, price='high')
	l_slopes, l_intercepts = calc_trendline(candlesticks, length, price='low')

	print(f'slope[high] : {np.sign(h_slopes[src])}')
	print(f'slope[low ] : {np.sign(l_slopes[src])}')

	#for Output
	slopes = {
		'high': {k : v for k, v in h_slopes.items() if src == k},
	 	'low' : {k : v for k, v in l_slopes.items() if src == k}
	}

	intercepts = {
		'high': {k : v for k, v in h_intercepts.items() if src == k}, 
		'low' : {k : v for k, v in l_intercepts.items() if src == k}
	}
	output_trendline(src, candlesticks, slopes, intercepts)

	if np.sign(h_slopes[src]) != np.sign(l_slopes[src]):
		print('Unstable')
		raise TypeError('is_rise_with_trendline : Return no boolean value(i.e. None)')

	if h_slopes[src] > 0 and l_slopes[src] > 0:
		return True
	else:
		return False

def calc_trendline(candlesticks, length, price='high'):
	slopes = {}
	intercepts = {}
	ohlc = {}
	for k, v in candlesticks.items():
		#ohlc[k] = v.ohlc.copy()[-length:]
		ohlc[k] = v.ohlc.copy()
		ohlc[k] = (ohlc[k] - ohlc[k].min()) / (ohlc[k].max() - ohlc[k].min())
		ohlc[k]['time_id'] = np.array([i+1 for i in range(len(ohlc[k]))])
		while len(ohlc[k]) > 3:
			x = ohlc[k]['time_id']
			y = ohlc[k][price]
			slopes[k], intercepts[k], _, _, _ = linregress(x, y)

			if price == 'high':
				left_hand = ohlc[k][price]
				right_hand = slopes[k] * x + intercepts[k]
			elif price == 'low':
				left_hand = slopes[k] * x + intercepts[k]
				right_hand = ohlc[k][price]
			else:
				print('input illegal parameter in price. only high or low')

			ohlc[k] = ohlc[k].loc[left_hand > right_hand]
	return slopes, intercepts

def output_trendline(src, candlesticks, slopes, intercepts):
	#Output
	#timeframes = list(candlesticks.keys())
	plothelper = PlotHelper()
	plothelper.begin_plotter()
	plothelper.add_candlestick(candlesticks)
	plothelper.add_double_trendline(candlesticks, slopes, intercepts)
	plothelper.end_plotter('trendline.png', True)
