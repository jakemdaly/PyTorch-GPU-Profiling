import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pdb

cuda = torch.device('cuda')
cpu = torch.device('cpu')

def main():

	# Get input parameters
	parser = argparse.ArgumentParser(description='TorchDCNN for deployment on GPU')
	parser.add_argument('network', help="'mnist' or 'celeba'")
	parser.add_argument('layer', help='which layer to send to the gpu')
	parser.add_argument('batch-size', help='give a batch size to profile')
	# parser.add_argument('vary-layer-dimensions', help='True or False')
	parser.add_argument('input-channel-depth', help='how many input channels the network will have')
	parser.add_argument('output-channel-depth', help='how many output channels the network will have')
	parser.add_argument('K', help='how big the Kernel will be (this will be used as width and height)')

	# Convert to Python variables
	args = parser.parse_args()
	variables = vars(args)
	batchSize = variables['batch-size']
	layer = variables['layer']
	network = variables['network']
	# varyLayerDim = variables['vary-layer-dimensions']
	numInputChannels = variables['input-channel-depth']
	numOutputChannels = variables['output-channel-depth']
	K = variables['K']


	# Convert to numeric if needed
	# if (varyLayerDim in ('True', 'T', 'true', 't')):
	# 	varyLayerDim = True
	# elif (varyLayerDim in ('False', 'F', 'false', 'f')):
	# 	varyLayerDim = False
	# else:
	# 	print("Invalid argument for vary-layer-dimensions. Must be true or false")
	# 	input()
	# 	exit()
	if (isinstance(numInputChannels, str)):
		numInputChannels = int(numInputChannels)
	if (isinstance(numOutputChannels, str)):
		numOutputChannels = int(numOutputChannels)    
	if (isinstance(K, str)):
		K = int(K)
	if (isinstance(batchSize, str)):
		batchSize = int(batchSize)
	if (not isinstance(batchSize, int)):
		print("error parsing batchsize")
		input()
		exit()
	if (isinstance(layer, str)):
		layer = int(layer)
	if (not isinstance(layer, int)):
		print("error parsing layer parameter")
		input()
		exit()
	
	print("[TorchDCNN]\tNetwork: %s\tLayer: %s\tBatch-Size: %s\tInputChDim: %s\tOutputChDim: %s\tK: %s"%(network, layer, batchSize, numInputChannels, numOutputChannels, K))
	
	seed = torch.manual_seed(0)

	if network=='mnist':
		net = mnist(layer, numInputChannels, numOutputChannels, K)
		if layer==1:
			inp = torch.rand(batchSize, numInputChannels,1,1)
		else:
			inp = torch.rand(batchSize, 10,1,1)

	elif network=='celeba':
		net = celeba(layer, numInputChannels, numOutputChannels, K)
		if layer==1:
			inp = torch.rand(batchSize, numInputChannels, 1, 1)
		else:
			inp = torch.rand(batchSize, 128,1,1)

	elif network=='pgan':
		net = pgan(layer, varyLayerDim, numInputChannels, numOutputChannels, K)
		if layer==1:
			inp = torch.rand(batchSize, numInputChannels, 4, 4)
		else:
			inp = torch.rand(batchSize, 512,4,4)

	else:
		print('Illegal network argument')
		assert(False)
	image = net(inp)


class mnist(nn.Module):

	def __init__(self, layerGPU, numInputChannels, numOutputChannels, K):
		super(mnist, self).__init__()
		self.layerGPU = layerGPU
		self.numInputChannels = numInputChannels
		self.numOutputChannels = numOutputChannels
		self.K = K

		assert(layerGPU <= 3 and layerGPU >= 1)
		assert(isinstance(layerGPU, int))

		if (self.layerGPU==1):
			self.layer1 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K), stride=2, padding=0).to(device=cuda)
			self.activation1 = nn.BatchNorm2d(self.numOutputChannels)
			self.ch = self.numOutputChannels # ch --> what the number of input channels for the next layer will be
		else:
			self.layer1 = nn.ConvTranspose2d(in_channels=10, out_channels=32, kernel_size=(4,4), stride=2, padding=0)
			self.activation1 = nn.BatchNorm2d(32)
			self.ch = 32 # ch --> what the number of input channels for the next layer will be

		self.out1 = nn.ReLU()
		
		if (self.layerGPU==2):
			self.layer2 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K),stride=2, padding=0).to(device=cuda)
			self.activation2 = nn.BatchNorm2d(self.numOutputChannels)
			self.ch=self.numOutputChannels # ch --> what the number of input channels for the next layer will be
		else:
			self.layer2 = nn.ConvTranspose2d(in_channels=self.ch, out_channels=32, kernel_size=(6,6),stride=2, padding=0)
			self.activation2 = nn.BatchNorm2d(32)
			self.ch=32 # ch --> what the number of input channels for the next layer will be

		self.out2 = nn.ReLU()

		if (self.layerGPU==3):
			self.layer3 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K), stride=2, padding=0).to(device=cuda)
		else:
			self.layer3 = nn.ConvTranspose2d(in_channels=self.ch, out_channels=1, kernel_size=(6,6), stride=2, padding=0)

		self.activation3 = nn.Sigmoid()

		for m in self.modules():
			if isinstance(m, nn.ConvTranspose2d):
				nn.init.normal_(m.weight)


	def forward(self, x):

		if (self.layerGPU==1):
			out1_ = self.layer1(x.to(device=cuda))
			out1_ = self.activation1(out1_.to(device=cpu))
		else:
			out1_ = self.layer1(x)
			out1_ = self.activation1(out1_)

		out1 = self.out1(out1_)
		
		if (self.layerGPU==2):
			out2_ = self.layer2(out1.to(device=cuda))
			out2_ = self.activation2(out2_.to(device=cpu))#.to(device=cpu)
		else:
			out2_ = self.layer2(out1)
			out2_ = self.activation2(out2_)

		out2 = self.out2(out2_)

		if (self.layerGPU==3):
			out3_ = self.layer3(out2.to(device=cuda))
			out3 = self.activation3(out3_.to(device=cpu))
		else:
			out3_ = self.layer3(out2)
			out3 = self.activation3(out3_)
		
		return(out3.detach().numpy())

class celeba(nn.Module):

	def __init__(self, layerGPU, numInputChannels, numOutputChannels, K):
		super(celeba, self).__init__()
		self.layerGPU = layerGPU
		self.numInputChannels = numInputChannels
		self.numOutputChannels = numOutputChannels
		self.K = K

		assert(layerGPU <= 5 and layerGPU >= 1)
		assert(isinstance(layerGPU, int))

		# Layer 1 ConvTranspose
		if (self.layerGPU==1):
			self.layer1 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K), stride=1, padding=0).to(device=cuda)
			self.activation1 = nn.BatchNorm2d(self.numOutputChannels)
			self.ch = self.numOutputChannels # ch --> what the number of input channels for the next layer will be
		else:
			self.layer1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=0)
			self.activation1 = nn.BatchNorm2d(128)
			self.ch = 128 # ch --> what the number of input channels for the next layer will be
		
		# Layer 1 RELU
		self.out1 = nn.ReLU()
		
		# Layer 2 ConvTranspose
		if (self.layerGPU==2):
			self.layer2 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K),stride=1, padding=0).to(device=cuda)
			self.activation2 = nn.BatchNorm2d(self.numOutputChannels)
			self.ch = self.numOutputChannels # ch --> what the number of input channels for the next layer will be
		else:
			self.layer2 = nn.ConvTranspose2d(in_channels=self.ch, out_channels=128, kernel_size=(3,3),stride=1, padding=0)
			self.activation2 = nn.BatchNorm2d(128)
			self.ch = 128 # ch --> what the number of input channels for the next layer will be
		
		# Layer 2 RELU
		self.out2 = nn.ReLU()

		# Layer 3 ConvTranspose
		if (self.layerGPU==3):
			self.layer3 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K),stride=1, padding=0).to(device=cuda)
			self.activation3 = nn.BatchNorm2d(self.numOutputChannels)
			self.ch = self.numOutputChannels # ch --> what the number of input channels for the next layer will be
		else:
			self.layer3 = nn.ConvTranspose2d(in_channels=self.ch, out_channels=64, kernel_size=(5,5),stride=1, padding=0)
			self.activation3 = nn.BatchNorm2d(64)
			self.ch = 64 # ch --> what the number of input channels for the next layer will be

		# Layer 3 RELU
		self.out3 = nn.ReLU()

		# Layer 4 ConvTranspose
		if (self.layerGPU==4):
			self.layer4 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K),stride=2, padding=0).to(device=cuda)
			self.activation4 = nn.BatchNorm2d(self.numOutputChannels)
			self.ch = self.numOutputChannels # ch --> what the number of input channels for the next layer will be
		else:
			self.layer4 = nn.ConvTranspose2d(in_channels=self.ch, out_channels=32, kernel_size=(5,5),stride=2, padding=0)
			self.activation4 = nn.BatchNorm2d(32)
			self.ch = 32 # ch --> what the number of input channels for the next layer will be
		# Layer 4 NORM

		# Layer 4 RELU
		self.out4 = nn.ReLU()

		# Layer 5 ConvTranspose
		if (self.layerGPU==5):
			self.layer5 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K), stride=2, padding=0).to(device=cuda)
		else:
			self.layer5 = nn.ConvTranspose2d(in_channels=self.ch, out_channels=3, kernel_size=(5,5), stride=2, padding=0)

		# Layer 5 SIGMOID
		self.activation5 = nn.Sigmoid()

		for m in self.modules():
			if isinstance(m, nn.ConvTranspose2d):
				nn.init.normal_(m.weight)


	def forward(self, x):

		# Layer 1
		if (self.layerGPU==1):
			out1_ = self.layer1(x.to(device=cuda))
			out1_ = self.activation1(out1_.to(device=cpu))
		else:
			out1_ = self.layer1(x)
			out1_ = self.activation1(out1_)

		out1 = self.out1(out1_)
		

		# Layer 2
		if (self.layerGPU==2):
			out2_ = self.layer2(out1.to(device=cuda))
			out2_ = self.activation2(out2_.to(device=cpu))#.to(device=cpu)
		else:
			out2_ = self.layer2(out1)
			out2_ = self.activation2(out2_)

		out2 = self.out2(out2_)


		# Layer 3
		if (self.layerGPU==3):
			out3_ = self.layer3(out2.to(device=cuda))
			out3_ = self.activation3(out3_.to(device=cpu))#.to(device=cpu)
		else:
			out3_ = self.layer3(out2)
			out3_ = self.activation3(out3_)

		out3 = self.out3(out3_)



		# Layer 4
		if (self.layerGPU==4):
			out4_ = self.layer4(out3.to(device=cuda))
			out4_ = self.activation4(out4_.to(device=cpu))#.to(device=cpu)
		else:
			out4_ = self.layer4(out3)
			out4_ = self.activation4(out4_)

		out4 = self.out4(out4_)


		# Layer 5
		if (self.layerGPU==5):
			out5_ = self.layer5(out4.to(device=cuda), output_size=(45,45))
			out5 = self.activation5(out5_.to(device=cpu))
		else:
			out5_ = self.layer5(out4, output_size=(45,45))
			out5 = self.activation5(out5_)
		
		return(out5.detach().numpy())


class pgan(nn.Module):

	def __init__(self, layerGPU, varyLayerDim, numInputChannels, numOutputChannels, K):
		super(pgan, self).__init__()
		self.layerGPU = layerGPU
		self.numInputChannels = numInputChannels
		self.numOutputChannels = numOutputChannels
		self.K = K
		self.varyLayerDim = varyLayerDim

		assert(layerGPU <= 6 and layerGPU >= 1)
		assert(isinstance(layerGPU, int))

		# Layer 1 ConvTranspose
		if (self.layerGPU==1 and self.varyLayerDim==True):
			self.layer1 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K), stride=2, padding=1).to(device=cuda) 
			self.activation1 = nn.BatchNorm2d(self.numOutputChannels)
		elif (self.layerGPU==1 and self.varyLayerDim==False):
			self.layer1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4,4), stride=2, padding=1).to(device=cuda)
			self.activation1 = nn.BatchNorm2d(512)
		else:
			self.layer1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4,4), stride=2, padding=1)
			self.activation1 = nn.BatchNorm2d(512)
		
		# Layer 1 RELU
		self.out1 = nn.ReLU()
		
		# Layer 2 ConvTranspose
		if (self.layerGPU==2 and self.varyLayerDim==True):
			self.layer2 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K),stride=2, padding=1).to(device=cuda)
			self.activation2 = nn.BatchNorm2d(self.numOutputChannels)
		elif (self.layerGPU==2 and self.varyLayerDim==False):
			self.layer2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4,4),stride=2, padding=1).to(device=cuda)
			self.activation2 = nn.BatchNorm2d(512)
		else:
			self.layer2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4,4),stride=2, padding=1)
			self.activation2 = nn.BatchNorm2d(512)
		
		# Layer 2 RELU
		self.out2 = nn.ReLU()

		# Layer 3 ConvTranspose
		if (self.layerGPU==3 and self.varyLayerDim==True):
			self.layer3 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K),stride=2, padding=1).to(device=cuda)
			self.activation3 = nn.BatchNorm2d(self.numOutputChannels)
		elif (self.layerGPU==3 and self.varyLayerDim==False):
			self.layer3 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4,4),stride=2, padding=1).to(device=cuda)
			self.activation3 = nn.BatchNorm2d(512)
		else:
			self.layer3 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4,4),stride=2, padding=1)
			self.activation3 = nn.BatchNorm2d(512)

		# Layer 3 RELU
		self.out3 = nn.ReLU()

		# Layer 4 ConvTranspose
		if (self.layerGPU==4 and self.varyLayerDim==True):
			self.layer4 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K),stride=2, padding=1).to(device=cuda)
			self.activation4 = nn.BatchNorm2d(self.numOutputChannels)
		elif (self.layerGPU==4 and self.varyLayerDim==False):
			self.layer4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4,4),stride=2, padding=1).to(device=cuda)
			self.activation4 = nn.BatchNorm2d(256)
		else:
			self.layer4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4,4),stride=2, padding=1)
			self.activation4 = nn.BatchNorm2d(256)
		# Layer 4 NORM

		# Layer 4 RELU
		self.out4 = nn.ReLU()

		# Layer 5 ConvTranspose
		if (self.layerGPU==5 and self.varyLayerDim==True):
			self.layer5 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K),stride=2, padding=1).to(device=cuda)
			self.activation5 = nn.BatchNorm2d(self.numOutputChannels)
		elif (self.layerGPU==5 and self.varyLayerDim==False):
			self.layer5 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4,4),stride=2, padding=1).to(device=cuda)
			self.activation5 = nn.BatchNorm2d(128)
		else:
			self.layer5 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4,4),stride=2, padding=1)
			self.activation5 = nn.BatchNorm2d(128)
		# Layer 5 NORM

		# Layer 5 RELU
		self.out5 = nn.ReLU()

		# Layer 6 ConvTranspose
		if (self.layerGPU==6 and self.varyLayerDim==True):
			self.layer6 = nn.ConvTranspose2d(in_channels=self.numInputChannels, out_channels=self.numOutputChannels, kernel_size=(self.K,self.K), stride=2, padding=1).to(device=cuda)
		elif (self.layerGPU==6 and self.varyLayerDim==False):
			self.layer6 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4,4), stride=2, padding=1).to(device=cuda)
		else:
			self.layer6 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4,4), stride=2, padding=1)

		# Layer 6 SIGMOID
		self.activation6 = nn.Sigmoid()

		for m in self.modules():
			if isinstance(m, nn.ConvTranspose2d):
				nn.init.normal_(m.weight)


	def forward(self, x):

		# Layer 1
		if (self.layerGPU==1):
			out1_ = self.layer1(x.to(device=cuda))
			out1_ = self.activation1(out1_.to(device=cpu))
		else:
			out1_ = self.layer1(x)
			out1_ = self.activation1(out1_)

		out1 = self.out1(out1_)
		
		# Layer 2
		if (self.layerGPU==2):
			out2_ = self.layer2(out1.to(device=cuda))
			out2_ = self.activation2(out2_.to(device=cpu))#.to(device=cpu)
		else:
			out2_ = self.layer2(out1)
			out2_ = self.activation2(out2_)

		out2 = self.out2(out2_)

		# Layer 3
		if (self.layerGPU==3):
			out3_ = self.layer3(out2.to(device=cuda))
			out3_ = self.activation3(out3_.to(device=cpu))#.to(device=cpu)
		else:
			out3_ = self.layer3(out2)
			out3_ = self.activation3(out3_)

		out3 = self.out3(out3_)

		# Layer 4
		if (self.layerGPU==4):
			out4_ = self.layer4(out3.to(device=cuda))
			out4_ = self.activation4(out4_.to(device=cpu))#.to(device=cpu)
		else:
			out4_ = self.layer4(out3)
			out4_ = self.activation4(out4_)

		out4 = self.out4(out4_)

		# Layer 5
		if (self.layerGPU==5):
			out5_ = self.layer5(out4.to(device=cuda))
			out5_ = self.activation5(out5_.to(device=cpu))#.to(device=cpu)
		else:
			out5_ = self.layer5(out4)
			out5_ = self.activation5(out5_)

		out5 = self.out5(out5_)

		# Layer 6
		if (self.layerGPU==6):
			out6_ = self.layer6(out5.to(device=cuda))
			out6 = self.activation6(out6_.to(device=cpu))
		else:
			out6_ = self.layer6(out5)
			out6 = self.activation6(out6_)
		
		return(out6.detach().numpy())


if __name__ == '__main__':

	main()
