import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pdb

cuda = torch.device('cuda')
cpu = torch.device('cpu')

def main():

    parser = argparse.ArgumentParser(description='TorchDCNN for deployment on GPU')
    parser.add_argument('network', help="'mnist' or 'celeba'")
    parser.add_argument('layer', help='which layer to send to the gpu')
    parser.add_argument('batch-size', help='give a batch size to profile')
    # parser.add_argument('--channel-dims', help='give the channel dimensions of each layer as an array')
    args = parser.parse_args()
    variables = vars(args)
    batchSize = variables['batch-size']
    layer = variables['layer']
    network = variables['network']
    # channelDims = variables['channel-dims']
    # print(channelDims)


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
    
    print("[TorchDCNN] Running layer %s on GPU with batchSize = %s"%(layer, batchSize))
    
    seed = torch.manual_seed(0)

    if network=='mnist':
        net = mnist(layer)
        if layer==1:
            inp = torch.rand(batchSize, 10,1,1).half()
        else:
            inp = torch.rand(batchSize, 10,1,1)
    elif network=='celeba':
        net = celeba(layer)
        if layer==1:
            inp = torch.rand(batchSize, 128, 1, 1).half()
        else:
            inp = torch.rand(batchSize, 128, 1, 1)
    else:
        print('Illegal network argument')
        assert(False)
    image = net(inp)



class mnist(nn.Module):

    def __init__(self, layerGPU):
        super(mnist, self).__init__()
        self.layerGPU = layerGPU

        assert(layerGPU <= 3 and layerGPU >= 1)
        assert(isinstance(layerGPU, int))

        if (self.layerGPU==1):
            self.layer1 = nn.ConvTranspose2d(in_channels=10, out_channels=32, kernel_size=(4,4), stride=2, padding=0).half().to(device=cuda)
        else:
            self.layer1 = nn.ConvTranspose2d(in_channels=10, out_channels=32, kernel_size=(4,4), stride=2, padding=0)

        self.activation1 = nn.BatchNorm2d(32)
        
        self.out1 = nn.ReLU()
        
        if (self.layerGPU==2):
            self.layer2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(6,6),stride=2, padding=0).half().to(device=cuda)
        else:
            self.layer2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(6,6),stride=2, padding=0)

        self.activation2 = nn.BatchNorm2d(32)
        
        self.out2 = nn.ReLU()

        if (self.layerGPU==3):
            self.layer3 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(6,6), stride=2, padding=0).half().to(device=cuda)
        else:
            self.layer3 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(6,6), stride=2, padding=0)

        self.activation3 = nn.Sigmoid()

        
        i = 1
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight)
                if i==self.layerGPU:
                    m.weight = m.weight.type(torch.half)
                    #print(m.weight.type())
                i += 1


    def forward(self, x):

        if (self.layerGPU==1):
            out1_ = self.layer1(x.to(device=cuda))
            out1_ = out1_.type(torch.float)
            #print(out1_.type())
            out1_ = self.activation1(out1_.to(device=cpu))
        else:
            out1_ = self.layer1(x)
            out1_ = self.activation1(out1_)

        out1 = self.out1(out1_)
        
        if (self.layerGPU==2):
            out2_ = self.layer2(out1.type(torch.half).to(device=cuda))
            out2_ = out2_.type(torch.float)
            out2_ = self.activation2(out2_.to(device=cpu))#.to(device=cpu)
        else:
            out2_ = self.layer2(out1)
            out2_ = self.activation2(out2_)

        out2 = self.out2(out2_)

        if (self.layerGPU==3):
            out3_ = self.layer3(out2.type(torch.half).to(device=cuda), output_size=(28,28))
            out3_ = out3_.type(torch.float)
            out3 = self.activation3(out3_.to(device=cpu))
        else:
            out3_ = self.layer3(out2, output_size=(28,28))
            out3 = self.activation3(out3_)
        
        return(out3.detach().numpy())

class celeba(nn.Module):

    def __init__(self, layerGPU):
        super(celeba, self).__init__()
        self.layerGPU = layerGPU

        assert(layerGPU <= 5 and layerGPU >= 1)
        assert(isinstance(layerGPU, int))

        # Layer 1 ConvTranspose
        if (self.layerGPU==1):
            self.layer1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=0).half().to(device=cuda)
        else:
            self.layer1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=0)

        # Layer 1 NORM
        self.activation1 = nn.BatchNorm2d(128)
        
        # Layer 1 RELU
        self.out1 = nn.ReLU()
        
        # Layer 2 ConvTranspose
        if (self.layerGPU==2):
            self.layer2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3,3),stride=1, padding=0).half().to(device=cuda)
        else:
            self.layer2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3,3),stride=1, padding=0)

        # Layer 2 NORM
        self.activation2 = nn.BatchNorm2d(128)
        
        # Layer 2 RELU
        self.out2 = nn.ReLU()

        # Layer 3 ConvTranspose
        if (self.layerGPU==3):
            self.layer3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5,5),stride=1, padding=0).half().to(device=cuda)
        else:
            self.layer3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5,5),stride=1, padding=0)

        # Layer 3 NORM
        self.activation3 = nn.BatchNorm2d(64)

        # Layer 3 RELU
        self.out3 = nn.ReLU()

        # Layer 4 ConvTranspose
        if (self.layerGPU==4):
            self.layer4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5,5),stride=2, padding=0).half().to(device=cuda)
        else:
            self.layer4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5,5),stride=2, padding=0)

        # Layer 4 NORM
        self.activation4 = nn.BatchNorm2d(32)

        # Layer 4 RELU
        self.out4 = nn.ReLU()

        # Layer 5 ConvTranspose
        if (self.layerGPU==5):
            self.layer5 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(5,5), stride=2, padding=0).half().to(device=cuda)
        else:
            self.layer5 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(5,5), stride=2, padding=0)

        # Layer 5 SIGMOID
        self.activation5 = nn.Sigmoid()


        i = 1
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight)
                if i==self.layerGPU:
                    m.weight = m.weight.type(torch.half)
                i += 1


    def forward(self, x):

        # Layer 1
        if (self.layerGPU==1):
            out1_ = self.layer1(x.to(device=cuda))
            out1_ = out1_.type(torch.float)
            out1_ = self.activation1(out1_.to(device=cpu))
        else:
            out1_ = self.layer1(x)
            out1_ = self.activation1(out1_)

        out1 = self.out1(out1_)
        

        # Layer 2
        if (self.layerGPU==2):
            out2_ = self.layer2(out1.type(torch.half).to(device=cuda))
            out2_ = out2_.type(torch.float)
            out2_ = self.activation2(out2_.to(device=cpu))#.to(device=cpu)
        else:
            out2_ = self.layer2(out1)
            out2_ = self.activation2(out2_)

        out2 = self.out2(out2_)


        # Layer 3
        if (self.layerGPU==3):
            out3_ = self.layer3(out2.type(torch.half).to(device=cuda))
            out3_ = out3_.type(torch.float)
            out3_ = self.activation3(out3_.to(device=cpu))#.to(device=cpu)
        else:
            out3_ = self.layer3(out2)
            out3_ = self.activation3(out3_)

        out3 = self.out3(out3_)



        # Layer 4
        if (self.layerGPU==4):
            out4_ = self.layer4(out3.type(torch.half).to(device=cuda))
            out4_ = out4_.type(torch.float)
            out4_ = self.activation4(out4_.to(device=cpu))#.to(device=cpu)
        else:
            out4_ = self.layer4(out3)
            out4_ = self.activation4(out4_)

        out4 = self.out4(out4_)


        # Layer 5
        if (self.layerGPU==5):
            out5_ = self.layer5(out4.type(torch.half).to(device=cuda), output_size=(45,45))
            out5_ = out5_.type(torch.float)
            out5 = self.activation5(out5_.to(device=cpu))
        else:
            out5_ = self.layer5(out4, output_size=(45,45))
            out5 = self.activation5(out5_)
        
        return(out5.detach().numpy())


if __name__ == '__main__':

    main()
