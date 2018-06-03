import json
import constante as const
import numpy as np

def loadWeigth():
	with open(const.FILE_WEIGTH, 'r') as file:
		weigth = json.loads(file.read())
		for key in weigth.keys():
			weigth[key] = np.matrix(weigth[key])
		return weigth

def loadTraining():
	with open(const.FILE_TRAINING, 'r') as file:
		training = json.loads(file.read())
		for i, data in enumerate(training):
			training[i]['input'] = np.array(data['input'])
		return training

def main():
	weigth = loadWeigth()
	training = loadTraining()

	print(weigth)
	print(training)

if __name__ == '__main__':
	main()
