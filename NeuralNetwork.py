import constante as const
import load
import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def training(weigth, ipt, opt):
	Theta1 = weigth['Theta1']
	Theta2 = weigth['Theta2']
	Theta2 = Theta2[:, 1:]

	ipt = np.c_[np.ones(ipt.shape[0]), ipt]
	opt = np.array([opt]).T

	for i in range(500):
		a1 = sigmoid(np.dot(ipt, Theta1.T))
		a2 = sigmoid(np.dot(a1, Theta2.T))
		del2 = (opt - a2) * (a2 * (1 - a2))
		del1


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
