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

def training(weigth, ipt, opt, fig_kit):
	ipt = np.c_[np.ones(ipt.shape[0]), ipt] # add bias value
	opt = opt.T # tranpose vector

	Theta1 = weigth['Theta1']
	Theta2 = weigth['Theta2']

	m = ipt.shape[0]
	cost_evolution = []
	arr = range(const.NB_BATCH)
	for i in arr:
		# feedForward
		a1 = sigmoid(np.dot(ipt, Theta1))
		a2 = sigmoid(np.dot(a1, Theta2))

		# cost function
		J = (1 / m) * sum(sum(-opt * np.log10(a2) - (1 - opt) * np.log10(1 - a2)))
		cost_evolution.append(J)
		# backpropagation
		delta_2 = (opt - a2) * (a2 * (1 - a2))
		delta_1 = (delta_2 * Theta2.T) * (a1 * (1 - a1))

		# update theta
		Theta2 += a1.T.dot(delta_2)
		Theta1 += ipt.T.dot(delta_1)

		print('\rprogress :', round(i * 100 / const.NB_BATCH), '% cost :', J, end='')
		if i % 250 == 0:
			updateGraph((cost_evolution, arr, i), (Theta1, Theta2), fig_kit)

	print('\ndone')
	ipt = ipt[:, 1:]
	for i in range(ipt.shape[0]):
		print(ipt[i], '>>', a2[i])

	weigth['Theta1'] = Theta1
	weigth['Theta2'] = Theta2

	return (cost_evolution, arr)

def feedForward(Theta1, Theta2, ipt):
	# add place for bias
	ipt = np.c_[np.ones(ipt.shape[0]), ipt]

	# feed forward
	a1 = sigmoid(np.dot(ipt, Theta1))
	a2 = sigmoid(np.dot(a1, Theta2))
	return a2

def createImg(size, theta):
	Theta1, Theta2 = theta

	A = np.linspace(0, 1, size)
	B = np.linspace(0, 1, size)
	img = []
	for va in A:
		ipt = np.c_[np.zeros(B.shape[0]), B]
		ipt[:, 0:1] = va
		img.append(feedForward(Theta1, Theta2, ipt).T[0])

	return np.array(img)

def updateGraph(cost, theta, fig_kit):
	# process parameters
	evo, arr, i = cost
	fig, ax1, ax2 = fig_kit

	# clear axis of first graph and draw updated plot
	ax1.clear()
	ax1.plot(arr[:i + 1], evo, color='black')

	img = createImg(const.SIZE_IMG, theta)
	# clear img and print the new one
	ax2.clear()
	ax2.imshow(img, cmap='gray', origin='lower')
	# draw image
	fig.canvas.draw()
	plt.pause(0.05)

def createGraph(cost, theta):
	evo, arr = cost
	img = createImg(const.SIZE_IMG, theta)

	plt.ioff()
	plt.figure('finale')

	plt.subplot(211)
	plt.plot(arr, evo, color='black')

	plt.subplot(212)
	plt.imshow(img, cmap='gray', origin='lower')

	plt.show()


def main():
	weigth = load.loadWeigth()
	ipt, opt = load.loadTraining()

	plt.ion()
	fig = plt.figure('training')
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	cost = training(weigth, ipt, opt, (fig, ax1, ax2))

	createGraph(cost, (weigth['Theta1'], weigth['Theta2']))

if __name__ == '__main__':
	main()
