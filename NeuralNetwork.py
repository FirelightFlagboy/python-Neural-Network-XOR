# coding: utf-8

import constante as const
import load
import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoidGradient(x):
	sig = sigmoid(x)
	return sig * (1 - sig).T

def training(weigth, ipt, opt):
	ipt = np.c_[np.ones(ipt.shape[0]), ipt] # add bias value
	opt = opt.T # tranpose vector

	Theta1 = weigth['Theta1'].T
	Theta2 = weigth['Theta2'].T

	m = ipt.shape[0]
	max_iter = 60000
	for i in range(60000):
		# feedForward
		a1 = sigmoid(np.dot(ipt, Theta1))
		a2 = sigmoid(np.dot(a1, Theta2))

		# cost function
		J = (1 / m) * sum(sum(-opt * np.log10(a2) - (1 - opt) * np.log10(1 - a2)))

		# backpropagation
		delta_2 = (opt - a2) * (a2 * (1 - a2))
		delta_1 = (delta_2 * Theta2.T) * (a1 * (1 - a1))

		# update theta
		Theta2 += a1.T.dot(delta_2)
		Theta1 += ipt.T.dot(delta_1)

		print('progress :', round(i * 100 / max_iter), '% cost :', J, end='\r')
	print('', a2, sep='\n')

def main():
	weigth = load.loadWeigth()
	ipt, opt = load.loadTraining()

	training(weigth, ipt, opt)
	# while True:
	# 	try:
	# 		plt.pause(1)
	# 	except KeyboardInterrupt:
	# 		plt.close('all')
	# 		break

if __name__ == '__main__':
	main()
