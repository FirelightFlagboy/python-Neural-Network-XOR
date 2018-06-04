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

def plotData():
	x = np.linspace(-5, 5, 100)
	y = sigmoid(x)

	fg = plt.figure(figsize=(10, 10))
	plt.subplot(221)
	plt.title('sigmoid')
	plt.plot(x, y, color='black')

	plt.draw()
	fg.show()
	return fg

def costFunction(Theta1, Theta2, X, y, lmb):
	m = X.shape[0]
	mT1 = Theta1.shape[0]
	mT2 = Theta2.shape[0]

	# feedForward(weigth, ipt):

	a1 = np.c_[np.ones(m), X]

	z2 = a1 * Theta1.T
	c = z2.shape[0]
	a2 = np.c_[np.ones(c), sigmoid(z2)]

	z3 = a2 * Theta2.T
	a3 = sigmoid(z3)

	# compute cost function
	p = (lmb / 2 * m) * (sum(sum(np.power(Theta1[:, 1:], 2)).T) + sum(sum(np.power(Theta2[:, 1:], 2)).T))
	J = (1 / m) * sum(sum(-y * np.log10(a3) - (1 - y) * np.log10(1 - a3))) + p
	J = J[0, 0]

	# compute backpropagation
	sig3 = (a3.T - y.T).T
	sig2 = (Theta2.T * sig3.T).T
	sig2 = sig2[:, 1:]

	del1 = sig2.T * a1
	del2 = sig3.T * a2

	Theta1_tmp = np.c_[np.zeros(mT1), Theta1[:, 1:]]
	Theta2_tmp = np.c_[np.zeros(mT2), Theta2[:, 1:]]

	grad1 = (1 / m) * del1 + (lmb / m) * Theta1_tmp
	grad2 = (1 / m) * del2 + (lmb / m) * Theta2_tmp
	return J, grad1, grad2

def updateWeigth(weigth, grad1, grad2):
	weigth['Theta1'] = weigth['Theta1'] - grad1
	weigth['Theta2'] = weigth['Theta2'] - grad2

def trainNeuralNetwork(weigth, ipt, opt, fig, ax1):
	arr = range(5000)
	cost = []
	for i in arr:
		try:
			# determine the cost and the gradient
			J, grad1, grad2 = costFunction(weigth['Theta1'], weigth['Theta2'], ipt, opt, const.LEARNING_RATE)

			# add cost to plot
			cost.append(J)
			ax1.clear()
			plt.subplot(222)
			plt.plot(arr[:i + 1], cost, color='black')
			plt.title('evolution of cost {}'.format(i + 1))
			plt.xlabel('iteration')
			plt.ylabel('cost J')

			print('iteration :', i, 'cost :', J)
			if cost.__len__() > 1:
				if J > cost[-2]:
					break
			# update the weigth
			updateWeigth(weigth, grad1, grad2)
			mapTroughAB(weigth)

			fig.canvas.draw()
			plt.pause(0.05)
		except KeyboardInterrupt:
			plt.close('all')
			break

def feedForward(weigth, ipt):
	Theta1 = weigth['Theta1']
	Theta2 = weigth['Theta2']

	a1 = np.c_[np.ones(ipt.shape[0]), ipt]
	z2 = a1 * Theta1.T
	a2 = np.c_[np.ones(z2.shape[0]), sigmoid(z2)]
	z3 = a2 * Theta2.T
	return sigmoid(z3)

def mapTroughAB(weigth):
	A = np.linspace(0, 1, 100)
	B = np.linspace(0, 1, 100)
	img = np.zeros((100, 100))

	for i, va in enumerate(A):
		ipt = np.c_[np.zeros(B.shape[0]), B]
		ipt[:, 0:1] = va
		res = feedForward(weigth, ipt)
		tt = np.reshape(res, 100)
		img[i] = tt

	# test = img.tolist()
	# print('look')
	# print('A = 0, B = 0', test[0][0])
	# print('A = 0, B = 1', test[0][99])
	# print('A = 1, B = 0', test[99][0])
	# print('A = 1, B = 1', test[99][99])

	plt.subplot(223)
	plt.imshow(img, cmap='gray', origin='lower', interpolation='nearest')
	plt.xlabel('A (in %)')
	plt.ylabel('B (in %)')
	plt.title('img')

def main():
	weigth = load.loadWeigth()
	ipt, opt = load.loadTraining()

	plt.ion()
	fig = plotData()
	ax1 = plt.subplot(222)

	trainNeuralNetwork(weigth, ipt, opt, fig, ax1)

	mapTroughAB(weigth)
	while True:
		try:
			plt.pause(1)
		except KeyboardInterrupt:
			plt.close('all')
			break

if __name__ == '__main__':
	main()
