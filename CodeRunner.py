#this code is to make the rest of the code a bit easier to use
from ConvolutNet import ConvolutNet
from UseConvolutNet import 	UseConvNet
def Run(UseOrTrain,ImagePath='./UseImages/00001.jpg', test=False):
	#runs this if training is chosen
	if UseOrTrain == 'train' or UseOrTrain == 'Train' or UseOrTrain == 'TRAIN':
		ConvolutNet(test)
	elif UseOrTrain == 'use' or UseOrTrain == 'Use' or UseOrTrain == 'USE':
		UseConvNet(ImagePath)
	else:
		print('please type "train" or "test"')
		Run(UseOrTrain, ImagePath=ImagePath,test=test)


print("Do you wish to 'train' or 'use'?")
TrainOrTest= input()
if TrainOrTest == 'train' or TrainOrTest == 'Train' or TrainOrTest == 'TRAIN':
	print("Do you wish to use testing mode? This will update you more often on the current progress, but it will be a bit slower. Type 'True' or 'False'")
	test = input()
	Run(TrainOrTest,test=test)

elif TrainOrTest == 'use' or TrainOrTest == 'Use' or TrainOrTest == 'USE':
	print("type in full image directory from current directory example './UseImages/00001.jpg' (must be .jpg file)")
	ImagePath = input()
	Run(TrainOrTest,ImagePath='./UseImages/00001.jpg')
