import json
import constante as const
import numpy as np

def initialise():
	weigth = {}
	weigth['Theta1'] = np.random.rand(const.NB_HIDDEN, const.NB_INPUT + 1).tolist()
	weigth['Theta2'] = np.random.rand(const.NB_OUPUT, const.NB_HIDDEN).tolist()
	return weigth

def main():
	weigth = initialise()
	string = json.dumps(weigth, indent=4, separators=(',', ': '))
	print(string)
	with open(const.FILE_WEIGTH, 'w') as file:
		file.write(string)

if __name__ == '__main__':
	main()
