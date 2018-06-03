import json
import constante as const
import numpy as np

def loadWeigth():
	with open(const.FILE_WEIGTH, 'r') as file:
		return json.loads(file.read())

def loadTraining():
	with open(const.FILE_TRAINING, 'r') as file:
		return json.loads(file.read())

def main():
	weigth = loadWeigth()
	training = loadTraining()

	print(weigth)
	print(training)

if __name__ == '__main__':
	main()
