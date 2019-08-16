import numpy as np
import pandas as pd
from scipy.stats import linregress
from decimal import *
import math

#User difined classes
from PlotHelper import PlotHelper

"""
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
"""

def is_rise_with_roundnr_line(src, candlesticks):
	#ざっくりとしたトレンドラインの傾きを調べる
	length = 2 ## fixme
	h_slopes, h_intercepts = calc_trendline(candlesticks, length, price='high')
	l_slopes, l_intercepts = calc_trendline(candlesticks, length, price='low')

	# トレンドラインの傾きがそろっていなければ例外
	if np.sign(h_slopes[src]) != np.sign(l_slopes[src]):
		print('Unstable')
		raise TypeError('is_rise_with_trendline : Return no boolean value(i.e. None)')

	"""
	try:
		# 上昇トレンドならば安値トレンドラインを採用
		if h_slopes[src] > 0 and l_slopes[src] > 0:
			slopes, intercepts = calc_trendline_maxarea(candlesticks, price='low')
		# 下降トレンドならば高値トレンドラインを採用
		else:
			slopes, intercepts = calc_trendline_maxarea(candlesticks, price='high')
	except ValueError as e:
		raise ValueError(f'{e}')
	"""

	# キリ番ラインの計算
	roundoffed, roundoffed_decimal, dist = calc_roundnr_line(src, candlesticks)

	#傾きが負なら安値キリ番ラインを使用,傾きが正なら高値キリ番ラインを使用
	if h_slopes[src] > 0 and l_slopes[src] > 0:
		roff = roundoffed['high']
		roff_deci = roundoffed_decimal['high']
		dist = dist['high']
	else:
		roff = roundoffed['low']
		roff_deci = roundoffed_decimal['low']
		dist = dist['low']

	"""
	#傾きが負なら高値キリ番ラインを使用,傾きが正なら安値キリ番ラインを使用
	if h_slopes[src] > 0 and l_slopes[src] > 0:
		roff = roundoffed['low']
		roff_deci = roundoffed_decimal['low']
		dist = dist['low']
	else:
		roff = roundoffed['high']
		roff_deci = roundoffed_decimal['high']
		dist = dist['high']
	"""

	#キリ番ラインの小数部が0.0or0.5の時に通過
	if not(roff_deci == 0.5 or roff_deci == 0.0):
		raise ValueError(f'is_rise_with_roundnr_line : weak round number line')
	#もしくは、キリ番ラインの小数部に応じて圧力を計算

	#反転可能性booleanで返す
	if h_slopes[src] < 0 and l_slopes[src] < 0:
		return True
	else:
		return False

def calc_roundnr_line(src, candlesticks):
	ohlc = {}
	ohlc[src] = candlesticks[src].ohlc.copy() # 時間足ごとのOHLCデータをコピー

	# 最も最近の高値or安値の小数部
	h_decimal = ohlc[src]['high'][-1] - math.floor(ohlc[src]['high'][-1])
	l_decimal = ohlc[src]['low'][-1] - math.floor(ohlc[src]['low'][-1])

	# 最も最近の高値or安値の整数部
	h_integer = math.floor(ohlc[src]['high'][-1])
	l_integer = math.floor(ohlc[src]['low'][-1])

	# Decimalモジュールで四捨五入を行うために数値を文字列にキャスト
	h_shifted_string = str(ohlc[src]['high'][-1] + 0.05)
	l_shifted_string = str(ohlc[src]['low'][-1] - 0.05)

	# Decimalモジュールを用いて小数第一位を四捨五入する
	h_roundoffed = float(Decimal(h_shifted_string).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
	l_roundoffed = float(Decimal(l_shifted_string).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))

	# 四捨五入後の数値の整数部
	h_roundoffed_integer = math.floor(h_roundoffed) 
	l_roundoffed_integer = math.floor(l_roundoffed)

	# Decimalモジュールを用いて四捨五入した値の小数部( => 0.5なら信頼度が強いライン)
	h_roundoffed_decimal = h_roundoffed - h_roundoffed_integer
	l_roundoffed_decimal = l_roundoffed - l_roundoffed_integer

	# キリ番ラインと高値or安値との距離を計算
	h_dist = np.abs(ohlc[src]['high'][-1] - h_roundoffed)
	l_dist = np.abs(ohlc[src]['low'][-1] - l_roundoffed)

	#ohlc[src] = (ohlc[src] - ohlc[src].min()) / (ohlc[src].max() - ohlc[k].min()) # OHLCデータの正規化

	# for Return
	roundoffed = {} # キリ番ラインの価格
	roundoffed_decimal = {} # キリ番ラインの小数部
	dist = {} # キリ番ラインと高値or安値との距離

	roundoffed['high'] = h_roundoffed
	roundoffed['low'] = l_roundoffed

	roundoffed_decimal['high'] = h_roundoffed_decimal
	roundoffed_decimal['low'] = l_roundoffed_decimal

	dist['high'] = h_dist
	dist['low'] = l_dist

	return roundoffed, roundoffed_decimal, dist


def is_rise_with_insidebar(src, candlesticks):
	ohlc = {}
	ohlc[src] = candlesticks[src].ohlc.copy() # 時間足ごとのOHLCデータをコピー

	# 高値と安値においてはらみ足になっていなければ例外
	if not (ohlc[src]['high'][-2] > ohlc[src]['high'][-1] and ohlc[src]['low'][-2] < ohlc[src]['low'][-1]):
		print('is_rise_with_insidebar : Unstable')
		raise ValueError('is_rise_with_insidebar : 2nd bar is lager than 1st bar in HIGH or LOW')

	"""
	# 始値と終値においてはらみ足になっていなければ例外
	if not((ohlc['open'][-2] > ohlc['open'][-1]) and (ohlc['close'][-2] < ohlc['close'][-1])):
		print('is_rise_with_harami : Unstable')
		raise ValueError('is_rise_with_insidebar : 2nd bar is lager than 1st bar in OPEN or CLOSE')
	"""

	# 最新の2本の足を比較し,n番目の足が(n-1)番目に比べて十分に短くない場合は例外
	if 0.5 * np.abs(ohlc[src]['open'][-2] - ohlc[src]['close'][-2]) < np.abs(ohlc[src]['open'][-1] - ohlc[src]['close'][-1]):
		print('is_rise_with_harami : Unstable')
		raise ValueError('is_rise_with_insidebar : The last bar is not short enough')

	output_insidebar(src, candlesticks) # 可視化

	# 始値と終値の差分の符号を計算。+なら反転して下降,-なら反転して上昇
	if np.sign(ohlc[src]['close'][-2] - ohlc[src]['open'][-2]) < 0:
		return True
	else:
		return False

def output_insidebar(src, candlesticks):
	extents = {}
	extents[src] = (len(candlesticks[src].ohlc) - 2, len(candlesticks[src].ohlc))

	#Output
	#timeframes = list(candlesticks.keys())
	plothelper = PlotHelper()
	plothelper.begin_plotter()
	plothelper.add_ohlc(candlesticks)
	plothelper.add_box(candlesticks, extents)
	plothelper.end_plotter('insidebar.png', True)

def is_rise_with_line_touch(src, candlesticks):
	threshold = 0.1 # const
	try:
		h_slopes, h_intercepts = calc_trendline_maxarea(candlesticks, price='high')
		l_slopes, l_intercepts = calc_trendline_maxarea(candlesticks, price='low')
	except ValueError as e:
		raise ValueError(f'{e}')

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

	x = len(candlesticks[src].ohlc) - 1 # 右端のx座標
	y = {} # 右端のx座標におけるトレンドラインの値
	price = {} # 右端の高値or安値
	diff = {} # 高値or安値の差分(y - price)

	# トレンドラインの右端の値
	y['high'] = h_slopes[src] * x + h_intercepts[src]
	y['low'] = l_slopes[src] * x + l_intercepts[src]

	# 右端の安値or高値
	price['high'] = candlesticks[src].ohlc['high'][-1]
	price['low'] = candlesticks[src].ohlc['low'][-1]

	# 高値or安値の差分(y - price)
	diff['high'] = y['high'] - price['high']
	diff['low'] = y['low'] - price['low']

	# トレンドラインの内側に高値or安値がない場合に例外
	if np.sign(diff['high']) < 0 or np.sign(diff['low']) > 0:
		raise ValueError('is_rise_with_line_touch : either of high price or low price is outside of trendline')

	# 高値トレンドラインと高値の距離が近い and 安値トレントラインと安値の距離が近い場合に例外
	if np.abs(diff['high']) < threshold and np.abs(diff['low']) < threshold:
		raise ValueError('is_rise_with_line_touch : both of high price and low price are close to trend line')

	# 高値トレンドラインと高値の距離が遠い and 安値トレントラインと安値の距離が遠い場合に例外
	if np.abs(diff['high']) >= threshold and np.abs(diff['low']) >= threshold:
		raise ValueError('is_rise_with_line_touch : both of high price and low price are far from trend line')

	if np.abs(y['low'] - price['low']) < threshold:
		return True
	else:
		return False

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

def is_rise_with_trendline_maxarea(src, candlesticks):
	try:
		h_slopes, h_intercepts = calc_trendline_maxarea(candlesticks, price='high')
		l_slopes, l_intercepts = calc_trendline_maxarea(candlesticks, price='low')
	except ValueError as e:
		raise ValueError(f'{e}')

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

def is_rise_with_trendline_maxarea(src, candlesticks):
	try:
		h_slopes, h_intercepts = calc_trendline_maxarea(candlesticks, price='high')
		l_slopes, l_intercepts = calc_trendline_maxarea(candlesticks, price='low')
	except ValueError as e:
		raise ValueError(f'{e}')

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

def calc_trendline_maxarea(candlesticks, price='high'):
	slopes = {}
	intercepts = {}
	ohlc = {}
	for k, v in candlesticks.items():
		ohlc[k] = v.ohlc.copy() # OHLCデータをコピー
		ohlc[k] = (ohlc[k] - ohlc[k].min()) / (ohlc[k].max() - ohlc[k].min()) # OHLCデータの正規化
		ohlc[k].set_index(np.array([i+1 for i in range(len(ohlc[k]))])) # 1スタートのインデックスを設定

		tmp_index = np.argmax(ohlc[k][price]) if price == 'high' else np.argmin(ohlc[k][price]) # 高値の最高値or安値の最安値のインデックスを取得
		if tmp_index != ohlc[k].index[0] # 高値の最高値or安値の最安値のインデックスが左端のインデックスと一致しているか比較
			raise ValueError('y[0] is not maximun or minimum')

		x = ohlc[k].index # X-axisにはインデックスを設定
		y = ohlc[k][price] # Y-axisには高値or安値を設定

		# slope_list, intercept_listはそれぞれt=0のデータを保持しない
		slope_list = [(y[i] - y[0]) / (x[i] - x[0]) for i in range(1, len(x)-1)] # x[0]を始点とした直線の傾きを計算
		intercept_list = [y[0] - slope * x[0] for slope in slope_list] # 傾きをもとに(x, y) = (x[0], y[0])を通る直線の切片を計算
		
		# sum_areaはt=0のデータを保持しない
		sum_area = np.array([1 for i in range(len(x))]) # 面積を格納する配列を生成((x,y)=(0,y[0])の面積は0より配列を省略)

		# y = slope[n] * x + intercept[n]
		# y = OHLC['high'][n] or OHLC['low'][n]
		# の各nにおける差分を計算->差分の総和を面積としてnごとに保存
		for n, slope, intercept in enumerate(zip(slope_list, intercept_list)): # 計算順序はトレンドラインが外側になるように設定
			 # 計算順序はトレンドラインが外側になるように設定
			sum_area[n] += sum([(slope*i+intercept) - y[i] if price == 'high' else y[i] - (slope*i+intercept) for i in range(1, len(x)-1)])

		# sum_area, slope_list, intercept_listはそれぞれt=0のデータを保持しない
		ind = np.argmax(sum_area) # 面積が最大となる傾きと切片を持つインデックスを取り出す
		slopes[k] = slope_list[ind]
		intercepts[k] = intercept_list[ind]

	return slopes, intercepts

"""
def calc_trendline3(candlesticks, price='high'):
	slopes = {}
	intercepts = {}
	ohlc = {}
	for k, v in candlesticks.items():
		ohlc[k] = v.ohlc.copy()
		ohlc[k] = (ohlc[k] - ohlc[k].min()) / (ohlc[k].max() - ohlc[k].min())
		ohlc[k].set_index(np.array([i+1 for i in range(len(ohlc[k]))]))

		#安値なら昇順　高値なら降順でsort
		ascending = True if price == 'low' else False

		tmp = ohlc[k].sort_values(price, ascending=ascending)
		#最小値がローソク足の0-5本目までに存在するか
		#->
		
		#sort後の上から3番までを参照
		tmp = tmp[:3]

		x = tmp.index
		y = tmp[price]

		is_in_order = [x[i] < x[i+1] for i in range(len(x) - 1)]
		if not all(is_in_order):
			raise ValueError('the order of x is invalid')

		slopes[k] = np.array([(y[i] - y[0]) / (x[i] - x[0]) for i in range(1, len(x))])

		if price == 'high':
			is_steep = [slopes[k][i] < slopes[k][-1] for i in range(len(slopes[k]) - 1)]
		else:
			is_steep = [slopes[k][i] > slopes[k][-1] for i in range(len(slopes[k]) - 1)]

		if any(is_steep):
			raise ValueError('middle slope is more steep')
		
		if price == 'high':
			slopes[k] = slopes[k].max()
		else:
			slopes[k] = slopes[k].min()

		intercepts[k] = y[0] - slopes[k] * x[0]

	return slopes, intercepts
"""

def output_trendline(src, candlesticks, slopes, intercepts):
	#Output
	#timeframes = list(candlesticks.keys())
	plothelper = PlotHelper()
	plothelper.begin_plotter()
	plothelper.add_ohlc(candlesticks)
	plothelper.add_double_trendline(candlesticks, slopes, intercepts)
	plothelper.end_plotter('trendline.png', True)
