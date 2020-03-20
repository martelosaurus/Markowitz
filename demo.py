class Dog:
	
	def __init__(self,name='Fido'):
		self.name = name

	def bark(self):
		print('ruff!')

	def hello(self):
		print('hello! my name is ' + self.name + '. ')
		self.bark()
	
