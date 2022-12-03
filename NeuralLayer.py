from Neuron import Neuron
import random

class NeuralLayer():
	def __init__(self, inputs, neurons, isInputLayer = False):
		self.__neurons = []
		self.isInputLayer = isInputLayer
		if isInputLayer:
			for i in range(neurons):
				self.__neurons.append(Neuron(weights = []))
		else:
			for n in range(neurons):	
				k = 0
				weights = [0] * inputs
				for j in range(inputs):
					weights[j] = random.random()
					k += weights[j]
				for j in range(inputs):
					weights[j] = weights[j] / k
				self.__neurons.append(Neuron(weights = weights))

	def __getitem__(self, key):
		return self.__neurons[key]
	
	def neuronsNum(self):
		return len(self.__neurons)

	def feedForward(self):
		for neuron in self.__neurons:
			neuron.feedForward()
			
	def correctWeights(self, loss, inputs, factor = 0.1):
		for n in range(len(self.__neurons)):
			self[n].correctWeights(loss, inputs, factor)
		
	# @property
 #    def attr(self):  
 #        return self.__attr

	# @attr.setter
 #    def attr(self, value):
 #        self.__attr = value
	
	