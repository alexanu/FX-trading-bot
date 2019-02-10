class RoutineInspector:
	def __init__(self, freq=10):
		self.CHECKING_FREQ = freq
		self.count = 0

	def is_inspect(self):
		if self.count == self.CHECKING_FREQ:
			return True
		else:
			return False

	def is_inspect_at_intervals_of(self, freq):
		if self.count == freq:
			return True
		else:
			return False
	
	def increment_count(self):
		self.count += 1
		return self.count

	def reset_count(self):
		self.count = 0
		return self.count

	def update(self):
		if self.is_inspect() is True:
			self.reset_count()
			return True
		else:
			self.increment_count()
			return False

	def update_at_intervals_of(self, freq):
		if self.is_inspect_at_intervals_of(freq) is True:
			self.reset_count()
			return True
		else:
			self.increment_count()
			return False
