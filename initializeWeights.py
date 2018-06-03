import json
import constante as const
import numpy as np

def main():
	weigth = {}
	weigth['Theta1'] = np.random.rand(const.NB_INPUT, const.NB_HIDDEN).tolist()
	weigth['Theta2'] = np.random.rand(const.NB_HIDDEN, const.NB_OUPUT).tolist()
	string = json.dumps(weigth, indent=4, separators=(',', ': '))
	print(string)
	with open(const.FILE_NAME, 'w') as file:
		file.write(string)

if __name__ == '__main__':
	main()
