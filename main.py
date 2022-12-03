import math 
from NeuralNetwork import NeuralNetwork

def mseLoss(xTrue, xPredicted):
	return -2 * (xTrue - xPredicted)
	
myNet = NeuralNetwork(inputs = 2, layers = [2, 2, 2, 2, 1])
tests = [
			[[-2, -1], 1], 	
			[[25, 6], 0],
			[[17, 4], 0],
			[[-15, -6], 1]
			# [[.1, .4], .5],
			# [[.4, .2], .6],
			# [[.3, .3], .6],
			# [[.4, .3], .7],
			# [[.5, .2], .7],
			# [[.1, .7], .8],
			# [[-.1, .1], .0],
			# [[-.2, .5], .3]
		]
for i in range(10):
	for test in tests:
		inputs = [x for x in test[0]]
		answer = test[1]
		outputs = myNet.feedForward(inputs)
		loss = mseLoss(answer, outputs[0])
		print("answer:", test[1], "output", round(outputs[0], 3), "loss:", round(loss, 3))
		myNet.correctWeights(loss, inputs, .25)

# print(round(myNet.feedForward([.0, .2])[0], 3))
# print(round(myNet.feedForward([.1, .3])[0], 3))
# print(round(myNet.feedForward([.9, .1])[0], 3))
# print(round(myNet.feedForward([-.1, .2])[0], 3))




	
