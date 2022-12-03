import math

class Neuron():
	def __init__(self, weights = [], bias = 0):
		self.weights = weights
		self.bias = bias
		self.value = 0
		self.childs = []
		self.parents = []
		self.accumBias = 0
		self.momentBias = 0.9
		self.accums = []
		self.moment = .9

	def appendChild(self, child):
		self.childs.append(child)

	def appendParent(self, parent):
		self.parents.append(parent)

	def feedForward(self):
		self.value = self.activate(self.calcSum())
		return self.value

	def calcSum(self):
		value = 0
		for i in range(len(self.childs)):
			value += self.childs[i].value * self.weights[i]
		return value + self.bias

	def correctWeights(self, loss, inputs, factor = 0.1):
		if len(self.accums) == 0:
			self.accums = [0 for i in range(len(self.childs))]
			
		dVal = self.activateDerivative(self.value, 1)
		acceleration = factor * dVal * loss
		self.accumBias = self.accumBias * self.momentBias - acceleration
		# self.bias += self.accumBias 
		self.bias -= acceleration
		for w in range(len(self.weights)):
			acceleration = factor * self.childs[w].value * dVal * loss
			self.accums[w] = self.accums[w] * self.moment - acceleration
			# self.weights[w] += self.accums[w]
			self.weights[w] -= acceleration
			self.childs[w].correctWeights(loss, inputs, factor)

	def activate(self, x, activationType = 1):
		if activationType == 0:   						# linear
			return x
		elif activationType == 1: 						# sigmoid
			return 1 / (1 + math.pow(math.e, -x))

	def activateDerivative(self, x, activationType = 1):
		if activationType == 0:   						# linear
			return 1
		elif activationType == 1:
			return x * (1 - x)