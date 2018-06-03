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
	ipt, opt = [], []
	with open(const.FILE_TRAINING, 'r') as file:
		training = json.loads(file.read())
		for data in training:
			ipt.append(data['input'])
			opt.append(data['output'])
		return np.matrix(ipt), np.array(opt)

def main():
	weigth = loadWeigth()
	training = loadTraining()

	ipt, opt = training
	print(weigth)
	print(ipt.shape, opt.shape)
	print(training)

if __name__ == '__main__':
	main()
