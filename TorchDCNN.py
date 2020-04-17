import torch
import torch.nn as nn
import torch.nn.functional as F
#import pdb
cuda = torch.device('cuda')
cpu = torch.device('cpu')

def main():

    seed = torch.manual_seed(0) 
    net = Generator()
    inp = torch.rand(1000, 10,1,1)#.unsqueeze(0)
    image = net(inp)
    #pdb.set_trace()
    #print(image.reshape(28,28))


class Generator(nn.Module):

    def __init__(self, ):
        super(Generator, self).__init__()
        self.z_dim = 10
        self.x_dim = 784
        self.name = 'mnist/dcgan/g_net'



        self.layer1 = nn.ConvTranspose2d(in_channels=10, out_channels=32, kernel_size=(4,4), stride=2, padding=0).to(device=cuda)
        
        self.activation1 = nn.BatchNorm2d(32)
        
        self.out1 = nn.ReLU()
        
        self.layer2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(6,6),stride=2, padding=0)
        
        self.activation2 = nn.BatchNorm2d(32)
        
        self.out2 = nn.ReLU()

        self.layer3 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(6,6), stride=2, padding=0)
        
        self.activation3 = nn.Sigmoid()

        

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight)


    def forward(self, x):

        out1_ = self.layer1(x.to(device=cuda))
        #pdb.set_trace()
        out1_ = self.activation1(out1_.to(device=cpu))
        out1 = self.out1(out1_)
        out2_ = self.layer2(out1)
        #pdb.set_trace()
        out2_ = self.activation2(out2_)
        out2 = self.out2(out2_)
        #pdb.set_trace()
        out3_ = self.layer3(out2, output_size=(28,28))
        #pdb.set_trace()
        out3 = self.activation3(out3_)

        return(out3.detach().numpy())

if __name__ == '__main__':

    main()
