import requests
import json
import sys

"""
Environment            Description
fxTrade(live)          The live(real money) environment
fxTrade Practice(demo) The Demo (virtual) environment
"""

streamDict = {
	'live':'stream-fxtrade.oanda.com',
	'demo':'stream-fxpractice.oanda.com'
}

apiDict = {
	'live':'api-fxtrade.oanda.com',
	'demo':'api-fxpractice.oanda.com'
}

class Endpoints(object):
	def __init__(self, environment='demo'):
		with open(sys.argv[1]) as f:
			auth_tokens = json.load(f)
			self._id = auth_tokens['oanda_id']
			self._token = auth_tokens['oanda_token']
			print(f'<OandaEndpoints> Read id and token from {sys.argv[1]}')

		self._api_domain = apiDict[environment]
		self._stream_domain = streamDict[environment]
		self._api_url = 'https://{}/v3/accounts/{}'.format(self._api_domain, self._id)
		self._stream_url = 'https://{}/v3/accounts/{}'.format(self._stream_domain, self._id)
		self._headers = {
			'Content-Type': 'application/json',
			'Authorization':'Bearer {}'.format(self._token)
		}


class Account(Endpoints):
	def __init__(self, environment='demo'):
		super().__init__(environment=environment)

	def get_list(self):
		url = 'https://{}/v3/accounts'.format(self._api_url)
		resp = requests.get(url, headers=self._headers)
		return resp

	def get_detail(self):
		resp = requests.get(self._api_url, headers=self._headers)
		return resp

	def get_summary(self):
		suffix = '/summary'
		resp = requests.get(self._api_url+suffix, headers=self._headers)
		return resp

	def get_tradable_instrument(self, instrument=None):
		suffix = '/instruments'
		if instrument is None:
			resp = requests.get(self._api_url+suffix, headers=self._headers)
		else:
			params = {
				'instruments':instrument
			}
			resp = requests.get(self._api_url+suffix, headers=self._headers, params=params)
		return resp

	def set_configuration(self, data):#data include margin rate etc...
		suffix = '/configuration'
		resp = requests.patch(self._api_url+suffix, headers=self._headers, data=json.dumps(data))
		return resp

	#よくわからない
	def change(self):
		pass


class Instrument(Endpoints):

	"""

	通貨ペアの情報を取り扱うクラス

	"""

	def __init__(self, environment='demo'):

		"""

		トレード環境を設定する

		Parameter
		---------
		environment : str
			トレード環境（本番or仮想）を指定

		"""

		super().__init__(environment=environment)
		self._api_url = 'https://{}/v3/instruments'.format(self._api_domain)

	def fetch_candle(self, instrument, price='M', granularity='S5', count=500):
		"""
		任意のローソク足を取得する

		Parameters
		----------
		instrument : 通貨ペア
		price : ['defualt='M'(mid point candle)], 'A'(ask candle), 'B'(bid candle)
		granularity : 時間足[default='S5']
		count : ローソク足の本数[default=500]
		"""
		suffix = '/{}/candles'.format(instrument)
		params = {
			'granularity': granularity,
			'count': count,
			'price': price
		}
		resp = requests.get(self._api_url+suffix, headers=self._headers, params=params)
		return resp

	#curlコマンドではcompressedオプションがついてたけど、requestsモジュールに反映させる方法がわからない
	def fetch_orderbook(self, instrument):
		suffix = '/{}/orderBook'.format(instrument)
		params = {
			'foo':'foo',
			'bar':'bar'
		}
		resp = requests.get(url, headers=self._headers, params=params)
		return resp

	#curlコマンドではcompressedオプションがついてたけど、requestsモジュールに反映させる方法がわからない
	def fetch_positionbook(self, instrument, params=None):
		suffix = '/{}/positionBook'.format(instrument)
		if params is None:
			resp = requests.get(url, headers=self._headers)
		else:
			resp = requests.get(url, headers=self._headers, params=params)
		return resp

class Order(Endpoints):
	def __init__(self, environment='demo'):
		super().__init__(environment=environment)

	def create_order(self, data):
		'''
		Examples(指値注文)
		--------
		data = {
			'order' : {
				'price' : '100.000',
				'instrument' : 'GBP_JPY',
				'units' : '+100',
				'type' : 'LIMIT',
				'positionFill' : 'DEFAULT'
			}
		}

		Examples(成行注文)
		--------
		data = {
			'order' : {
				'instrument' : 'GBP_JPY',
				'units' : '+100',
				'type' : 'MARKET',
				'positionFill' : 'DEFAULT'
			}
		}
		'''
		suffix = '/orders'
		resp = requests.post(self._api_url+suffix, headers=self._headers, data=json.dumps(data))
		return resp

	def get_list(self, instrument):
		suffix = '/orders'
		params = {
			'instrument':instrument
		}
		resp = requests.get(self._api_url+suffix, headers=self._headers, params=params)
		return resp

	def get_pending_list(self):
		suffix = '/pendingOrders'
		resp = requests.get(self._api_url+suffix, headers=self._headers)
		return resp

	def get_detail(self, order_id):
		suffix = '/orders/{}'.format(order_id)
		resp = requests.get(self._api_url+suffix, headers=self._headers)
		return resp

	def replace_order(self, data):
		suffix = '/orders/{}'.format(order_id)
		resp = requests.put(self._api_url+suffix, headers=self._headers, data=json.dumps(data))
		return resp
		

	#保留中(pending)の注文をorder_idをもとにキャンセルする
	def cancel_order(self, order_id):
		suffix = '/orders/{}/cancel'.format(order_id)
		resp = requests.put(self._api_url+suffix, headers=self._headers)
		return resp

	#実装予定なし
	def client_extension(self):
		pass

class Trade(Endpoints):
	def __init__(self, environment='demo'):
		super().__init__(environment=environment)

	def get_list(self, instrument):
		suffix = '/trades'
		params = {'instrument':instrument}
		resp = requests.get(self._api_url+suffix, headers=self._headers, params=params)
		return resp

	def get_open_trade(self):
		suffix = '/openTrades'
		resp = requests.get(self._api_url+suffix, headers=self._headers, params=params)
		return resp

	def get_detail(self, trade_id):
		suffix = '/trades/{}'.format(trade_id)
		resp = requests.get(self._api_url+suffix, headers=self._headers)
		return resp

	#trade_idをもとに決済する
	def close_trade(self, data, trade_id):
		suffix = '/trades/{}/close'.format(trade_id)
		resp = requests.put(self._api_url+suffix, headers=self._headers, data=json.dumps(data))
		return resp

	#実装予定なし
	def client_extension(self):
		pass

	#crc means Create Replace Cancel
	def crc_order(self, data, trade_id):
		suffix = '/trades/{}/orders'.format(trade_id)
		resp = requests.put(self._api_url+suffix, headers=self._headers, data=json.dumps(data))
		return resp

class Position(Endpoints):
	def __init__(self, environment='demo'):
		super().__init__(environment=environment)

	def get_list(self):
		suffix = '/positions'
		resp = requests.get(self._api_url+suffix, headers=self._headers)
		return resp

	def get_open_position(self):
		suffix = '/openPositions'
		resp = requests.get(self._api_url+suffix, headers=self._headers)
		return resp

	def get_detail(self, instrument):
		suffix = '/positions/{}'.format(instrument)
		resp = requests.get(self._api_url+suffix, headers=self._headers)
		return resp

	#保有しているポジションを決済する
	def close_position(self, data, instrument):
		suffix = '/positions/{}/close'.format(instrument)
		resp = requests.put(self._api_url+suffix, headers=self._headers, data=json.dumps(data))
		return resp

class Transaction(Endpoints):
	def __init__(self, environment='demo'):
		super().__init__(environment=environment)

	def get_list(self, src, dst):
		suffix = '/transactions'
		params = {
			'to':dst,
			'from':src
		}
		resp = requests.get(self._api_url+suffix, headers=self._headers, params=params)
		return resp

	def get_detail(self, transaction_id):
		url = 'https://{}/v3/accounts/{}/transactions/{}'.format(self._domain, self._id, transaction_id) 
		suffix = '/transactions/{}'.format(transaction_id)
		resp = requests.get(self._api_url+suffix, headers=self._headers)
		return resp

	def get_idrange(self, src_id, dst_id):
		suffix = '/transactions/idrange'
		params = {
			'to':dst_id,
			'from':src_id
		}
		resp = requests.get(self._api_url+suffix, headers=self._headers, params=params)
		return resp

	def get_sinceid(self, transaction_id):
		suffix = '/transactions/sinceid'
		params = {
			'id':transaction_id
		}
		resp = requests.get(self._api_url+suffix, headers=self._headers, params=params)
		return resp

	def connect_to_stream(self):
		suffix = '/transactions/stream'
		resp = requests.get(self._streamurl+suffix, headers=self._headers)
		return resp

class Pricing(Endpoints):
	def __init__(self, environment='demo'):
		super().__init__(environment=environment)

	def get_pricing_information(self, instrument):
		suffix = '/pricing'
		params = {
			'instruments':instrument
		}
		resp = requests.get(self._api_url+suffix, headers=self._headers, params=params)
		return resp

	def connect_to_stream(self, instrument):
		#not using session
		'''		
		try:
#			url = "https://" + self._domain + "/v3/accounts/" + account_id +"/pricing/stream"
#			url = 'https://{}/v3/accounts/{}/pricing/stream'.format(self._domain, self._id)
			suffix = '/pricing/stream'
			headers = {
				'Authorization':'Bearer {}'.format(self._token),
				#'X-Accept-Datetime-Format':'unix'
			}
			params = {
				'instruments':instrument
			}
			req = requests.get(self._stream_url+suffix, headers=headers, params=params, stream=True)
			return req
		except Exception as e:
			print("Caught exception when connecting to stream : {}".format(str(e)))

		'''
		#using session
		try:
			s = requests.Session()
#			url = "https://" + self._domain + "/v3/accounts/" + account_id +"/pricing/stream"
#			url = 'https://{}/v3/accounts/{}/pricing/stream'.format(self._stream, self._id)
			suffix = '/pricing/stream'
			headers = {#'Authorization':'Bearer ' + access_token, 
						'Authorization':'Bearer {}'.format(self._token),
						#'X-Accept-Datetime-Format':'unix'
						}
			params = {'instruments':instrument}
			req = requests.Request('GET', self._stream_url+suffix, headers=headers, params=params)
			pre = req.prepare()
			resp = s.send(pre, stream = True, verify = True)
			return resp
		except Exception as e:
			s.close()
			print("Caught exception when connecting to stream : {}".format(str(e)))

def from_byte_to_dict(byte_line):
	"""
	byte型の文字列をdict型に変換する

	Parameters
	----------
	byte_line: byte
		byte型の文字列
	"""

	try:
		return json.loads(byte_line.decode("UTF-8"))
	except Exception as e:
		print("Caught exception when converting message into json : {}" .format(str(e)))
		return None

def from_response_to_dict(response):
	try:
		return json.loads(response.content.decode('UTF-8'))
	except Exception as e:
		print("Caught exception when converting message into json : {}" .format(str(e)))
		return None

def test_pricing():
	endpoint = Pricing()
	response = endpoint.get_pricing_information('GBP_JPY')
	print(response.status_code)
	response = endpoint.connect_to_stream('GBP_JPY')
	print(response.status_code)
	if response.status_code != 200:
		print(response.text)
	for line in response.iter_lines():
		print(line)

def test_account():
	endpoint = Account()
	#response = endpoint.get_list()
	#print(response.status_code)
	response = endpoint.get_detail()
	print(response.status_code)
	response = endpoint.get_summary()
	print(response.status_code)
	response = endpoint.get_tradable_instrument()
	print(response.status_code)

def test_position():
	endpoint = Position()
	response = endpoint.get_list()
	print(response.status_code)
	response = endpoint.get_open_position()
	print(response.status_code)
	response = endpoint.get_detail('USD_JPY')
	print(response.status_code)

if __name__ == '__main__':
	pricing_obj = Pricing()
	order_obj  = Order()
	resp = pricing_obj.get_pricing_information('GBP_JPY')
	print(resp.status_code)
	print(resp.text)

	'''
	resp = order_obj.create_order(data=json.dumps(data))
	print(resp.status_code)
	print(resp.text)
	'''
	resp = order_obj.get_pending_list()
	print(resp.status_code)
	print(resp.text)

	resp = order_obj.cancel_order(order_id=12)
	print(resp.status_code)
	print(resp.text)

	
	resp = order_obj.get_pending_list()
	print(resp.status_code)
	print(resp.text)

	'''
	position_obj = Position()
	resp = position_obj.get_list()
	print(resp.status_code)
	print(resp.text)
	'''

	'''
	data = {
		'longUnits' : 'ALL'
	}
	resp = position_obj.close_position(data, instrument='EUR_USD')
	print(resp.status_code)
	print(resp.text)
	'''
