import numpy as np

from OandaEndpoints import Order, Position, Pricing, Instrument
from Plotter import Plotter

from Notify import notify_from_line
from OandaEndpoints import from_byte_to_dict, from_response_to_dict

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

	def set_order(self, kind, notify):
		if kind is not None:
			resp = self.pricing.get_pricing_information(self.instrument)
			msg = from_byte_to_dict(resp.content)
			self.pre_price = float(msg['prices'][0]['bids'][0]['price'])
			self.kind = kind
			if notify is True:
				notify_from_line(f'Order[{self.kind}] : {self.pre_price}')

	def set_close(self, notify):
		resp = self.pricing.get_pricing_information(self.instrument)
		msg = from_byte_to_dict(resp.content)
		self.post_price = float(msg['prices'][0]['bids'][0]['price'])
		if notify is True:
			notify_from_line(f'Close : {self.post_price}')

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
