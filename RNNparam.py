class RNNparam:
	def __init__(self, rate=None):
		#Shared param
		input_dim = 1
		output_dim = 1
		hidden_dims = [4]
		actvs = ['tanh']
		
		tau = None
		hidden_units = hidden_dims[0]
		keep_prob = 1.0

		#Uniqe param
		if rate == '1min':
			tau = 45
			t_form = ['average', 15]

		if rate == '5min':
			tau = 60
			t_form = ['average', 15]

		if rate == '15min':
			tau = 45
			t_form = ['average', 15]

		if rate == '4H':
			tau = 60
			t_form = ['average', 15]


		#Layer config
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden_dims = hidden_dims
		self.actvs = ['tanh']

		#Cell config
		self.tau = tau
		self.hidden_units = self.hidden_dims[0]
		self.keep_prob = keep_prob

		#Format of trainer data
		self.trainer_candle = {
			'preproc': 'norm',
			'x_form': None,
			't_form': t_form,
			'rate': rate,
			'price': 'close',
			'tau': tau
		}
