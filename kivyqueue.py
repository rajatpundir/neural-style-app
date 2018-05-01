from queue import Queue

class KivyQueue(Queue):
	# Multithread Safe Queue.
	callback_function = None

	def __init__(self, callback_function, **kwargs):
		super().__init__(self, **kwargs)
		self.maxsize = 0
		self.callback_function = callback_function

	def put(self, val):
		Queue.put(self, val, False)
		self.callback_function()

	def get(self):
		return Queue.get(self, False)

