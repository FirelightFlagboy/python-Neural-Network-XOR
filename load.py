import json
import constante as const
import numpy as np

def loadWeigth():
	with open(const.FILE_WEIGTH, 'r') as file:
		weigth = json.loads(file.read())
		for key in weigth.keys():
			weigth[key] = np.array(weigth[key]).T
		return weigth

def loadTraining():
	ipt, opt = [], []
	with open(const.FILE_TRAINING, 'r') as file:
		training = json.loads(file.read())
		for data in training:
			ipt.append(data['input'])
			opt.append(data['output'])
		return np.array(ipt), np.array([opt])

def saveWeigth(weigth):
	weigth['Theta1'] = weigth['Theta1'].tolist()
	weigth['Theta2'] = weigth['Theta2'].tolist()
	string = json.dumps(weigth, indent=4, separators=(',', ': '))
	with open(const.FILE_WEIGTH, 'w') as file:
		file.write(string)

def main():
	weigth = loadWeigth()
	training = loadTraining()

	ipt, opt = training
	print(weigth)
	print(ipt.shape, opt.shape)
	print(training)
	saveWeigth(weigth)

if __name__ == '__main__':
	main()
