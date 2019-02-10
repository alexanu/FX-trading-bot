class Trader:
	def __init__(self, instrument, environment='demo', mode='test'):
		self.state = None
		self.mode = mode
		self.instrument = instrument
		self.order = Order(environment)
		self.position = Position(environment)

		#for debug
		self.count = 0

	def arrange_data_for_limit(self, kind, instrument, units, price):
		price = '{}'.format(price)
		sign = '+' if kind is 'BUY' else '-'
		units = '{}{}'.format(sign, units)
		data = {
			'order': {
				'price': price,
				'instrument': instrument,
				'units':units,
				'type': 'LIMIT',
				'positionFill': 'DEFAULT'
			}
		}
		return data

	def arrange_data_for_market(self, kind, instrument, units):
		sign = '+' if kind is 'BUY' else '-'
		units = '{}{}'.format(sign, units)
		data = {
			'order': {
				'instrument': instrument,
				'units':units,
				'type': 'MARKET',
				'positionFill': 'DEFAULT'
			}
		}
		return data

	def test_create_order(self, is_rise=None):
		if is_rise is True:
			debug_print('BUY')
		elif is_rise is False:
			debug_print('SELL')
		else:
			return False
				
		return True

	def create_order(self, is_rise=None):
		if is_rise is True:
			if self.mode == 'test':
				debug_print('BUY')
			else:
				self.market_order('BUY', self.instrument, 10)
				time.sleep(1)
				print('create order BUY')
		elif is_rise is False:
			if self.mode == 'test':
				debug_print('SELL')
			else:
				self.market_order('SELL', self.instrument, 10)
				time.sleep(1)
				print('create order SELL')
		else:
			return False
				
		return True

	def test_close_position(self):
		#for debug
		debug_print('Close position')
		return True

	def close_position(self):
		if self.mode == 'test':
			pass
		else:
			resp = self.position.get_open_position()
			time.sleep(1)
			pos_dic = from_response_to_dict(resp)
			for i in range(len(pos_dic['positions'])):
				instrument = pos_dic['positions'][i]['instrument']
				for order_type in ['long','short']:
					if pos_dic['positions'][i][order_type]['units'] is not '0':
						data = {
							'{}Units'.format(order_type): 'ALL'
						}
						self.position.close_position(data, instrument)
						time.sleep(1)

		print('Close position')
		return True

	def can_close_position(self):
		threshold = 1
		if self.count > threshold:
			return True
		else:
			return False

	def update_whether_closing_position(self):
		if self.can_close_position() is True:
			self.count = 0
		else:
			self.count += 1
	
	def test_is_reflected_order(self):
		if self.state == 'WAIT_ORDER':
			resp = self.order.get_pending_list()
			order_dic = from_response_to_dict(resp)
			time.sleep(1)
			
			#empty order for debug
			order_dic = {
				'orders': []
			}

			if not self.has_order(order_dic) is True:
				return True
		return False

	def is_reflected_order(self):
		if self.state == 'WAIT_ORDER':
			resp = self.order.get_pending_list()
			order_dic = from_response_to_dict(resp)
			time.sleep(1)
			
			if self.mode == 'test':
				#empty order for debug
				order_dic = {
					'orders': []
				}
			else:
				pass

			if not self.has_order(order_dic) is True:
				return True
		return False

	def test_is_reflected_position(self):
		if self.state == 'WAIT_POSITION':
			resp = self.position.get_open_position()
			pos_dic = from_response_to_dict(resp)
			time.sleep(1)

			#empty position for debug
			pos_dic = {
				'positions': []
			}

			if not self.has_position(pos_dic) is True:
				return True
		return False

	def is_reflected_position(self):
		if self.state == 'WAIT_POSITION':
			resp = self.position.get_open_position()
			pos_dic = from_response_to_dict(resp)
			time.sleep(1)

			if self.mode == 'test':
				#empty position for debug
				pos_dic = {
					'positions': []
				}
			else:
				pass

			if not self.has_position(pos_dic) is True:
				return True
		return False

	#保留中の注文と保有中のポジションを空にする
	def clean(self):
		resp = self.order.get_pending_list()
		time.sleep(1)
		order_dic = from_response_to_dict(resp)

		resp = self.position.get_open_position()
		time.sleep(1)
		pos_dic = from_response_to_dict(resp)

		
		#保留中の注文をすべてキャンセルする
		if self.has_order(order_dic) is True:
			for i in range(len(order_dic['orders'])):
				order_id = order_dic['orders'][i]['id']
				self.order.cancel_order(order_id)
				time.sleep(1)
			debug_print('cancel order')

		#Oandaのサーバー上に反映されるまで待つ
		while self.has_order(order_dic) is True:
			resp = self.order.get_pending_list()
			time.sleep(1)
			order_dic = from_response_to_dict(resp)
			has_odr = 'exist' if self.has_order(order_dic) is True else 'clean'
			print('pending order: {}'.format(has_odr))

		#保有しているポジションをすべて決済する
		if self.has_position(pos_dic) is True:
			for i in range(len(pos_dic['positions'])):
				instrument = pos_dic['positions'][i]['instrument']
				for order_type in ['long','short']:
					if pos_dic['positions'][i][order_type]['units'] is not '0':
						data = {
							'{}Units'.format(order_type): 'ALL'
						}
						self.position.close_position(data, instrument)
						time.sleep(1)
			debug_print('close positon')

		#Oandaのサーバー上に反映されるまで待つ
		while self.has_position(pos_dic) is True:
			resp = self.position.get_open_position()
			time.sleep(1)
			pos_dic = from_response_to_dict(resp)
			has_pos = 'exist' if self.has_position(pos_dic) is True else 'clean'
			print('position: {}'.format(has_pos))

		#ORDERモードに状態遷移する
		self.state = 'ORDER'
		print(self.state)
		return self.state

	def execute(self):
		if self.state == 'ORDER':
			has_ord = self.exec_order()
			#+limit order
			#+stop order
			if has_ord is True:
				self.switch_state()

		elif self.state == 'POSITION':
			has_pos = self.exec_position()
			if has_pos is True:
				self.switch_state()

		else:
			pass
			
		if self.is_reflected_order() is True:
			self.switch_state()
		if self.is_reflected_position() is True:
			self.switch_state()
		print(self.state)
		return self.state

	def exec_order(self, is_rise=None):
		if is_rise is True:
			self.market_order('BUY', self.instrument, 10)
			time.sleep(1)
			print('create order BUY')

			#for evaluate
			order_kind = 'Buy'
		else:
			self.market_order('SELL', self.instrument, 10)
			time.sleep(1)
			print('create order SELL')

			#for evaluate
			order_kind = 'Sell'

		return order_kind

	def exec_position(self):
		self.counter += 1
		threshold = 4
		print(f'counter: {self.counter} threshold: {threshold}')
		if self.counter > threshold:
			resp = self.position.get_open_position()
			time.sleep(1)
			pos_dic = from_response_to_dict(resp)
			for i in range(len(pos_dic['positions'])):
				instrument = pos_dic['positions'][i]['instrument']
				for order_type in ['long','short']:
					if pos_dic['positions'][i][order_type]['units'] is not '0':
						data = {
							'{}Units'.format(order_type): 'ALL'
						}
						self.position.close_position(data, instrument)
						time.sleep(1)

			print('close position')
			#reset conter
			self.counter = 0

			return True
		else:
			return False

	def has_order(self, order_dic):
		if bool(order_dic['orders']) is True:
			return True
		else:
			return False

	def has_position(self, pos_dic):
		if bool(pos_dic['positions']) is True:
			return True
		else:
			return False

	#指値注文
	def limit_order(self, kind, instrument, units, price):
		data = self.arrange_data_for_limit(kind, instrument, units, price)					
		resp = self.order.create_order(data)
		#for debug
		print(resp.text)

		#stateは変化させない
		#self.state = 'WAIT_ORDER'

	#成行注文
	def market_order(self, kind, instrument, units):
		data = self.arrange_data_for_market(kind, instrument, units)					
		resp = self.order.create_order(data)
		#for debug
		print(resp.text)

		#stateは変化させない
		#self.state = 'WAIT_ORDER'

	def switch_state(self):
		if self.state == 'ORDER':
			self.state = 'WAIT_ORDER'

		elif self.state == 'WAIT_ORDER':
			self.state = 'POSITION'

		elif self.state == 'POSITION':
			self.state = 'WAIT_POSITION'

		elif self.state == 'WAIT_POSITION':
			self.state = 'ORDER'
		
		else:
			pass
