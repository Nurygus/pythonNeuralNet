from NeuralLayer import NeuralLayer
import random

class NeuralNetwork():
	def __init__(self, inputs = 1, layers = [1, 1]):
		self.layers = []
		self.layers.append(NeuralLayer(inputs = inputs, neurons = inputs, isInputLayer = True))
		for layer in layers:
			self.layers.append(NeuralLayer(inputs = inputs, neurons = layer))
			
			preL = self.layers[-2]
			curL = self.layers[-1]
			for i in range(preL.neuronsNum()):
				for j in range(curL.neuronsNum()):
					curL[j].appendChild(preL[i])
					preL[i].appendParent(curL[j])
			inputs = layer 

	def feedForward(self, values):
		for i in range(self.layers[0].neuronsNum()):
			self.layers[0][i].value = values[i]

		for i in range(1, len(self.layers)):
			self.layers[i].feedForward()
			
		return [neuron.value for neuron in self.layers[-1]]

	def correctWeights(self, loss, inputs, factor = 0.1):
		self.layers[-1].correctWeights(loss, inputs, factor)



