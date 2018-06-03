import constante as const
import load
import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def costFunction(Theta1, Theta2, X, y, lReate):
	pass

def main():
	x = np.linspace(-5, 5, 100)
	y = sigmoid(x)

	plt.figure()
	plt.subplot(211)
	plt.title('sigmoid')
	plt.plot(x, y, color='black')

	plt.show()

if __name__ == '__main__':
	main()
