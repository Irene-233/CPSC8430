#Deep Convolution Generative Adversarial Networks

#Importing the Libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.optim import RMSprop
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import fid as FID

#Setting some hyperparameters
batchSize = 64
imageSize = 64

#Creating the transformations
transform = transforms.Compose([transforms.Resize(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

#Loading the dataset 
dataset = dset.CIFAR10(root = './data', download = True, transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2)

#Defining the weights_init function that takes as input a neural net and initializes its weights

def weights_init(m):
    # weight_initialization: important for wgan
    class_name=m.__class__.__name__
    if class_name.find('Conv')!=-1:
        m.weight.data.normal_(0,0.02)
    elif class_name.find('Norm')!=-1:
        m.weight.data.normal_(1.0,0.02)
#     else:print(class_name)        
#Defining the generator

class G(nn.Module):
    
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
            nn.Tanh()
        )
    
    def forward(self, input):
        output = self.main(input)
        return output

#Creating the generator
netG = G()
netG.apply(weights_init)

#Defining the discriminator

class D(nn.Module):
    
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 1, 4, 1, 0, bias = False),
            #nn.Sigmoid()
        )
    
    def forward(self, input):
        output = self.main(input)
        return output.view(-1)
    
#Creating the discriminator
netD = D()
netD.apply(weights_init)

#Training the DCGAN's

#criterion = nn.BCELoss()
optimizerD = RMSprop(netD.parameters(),lr=0.00005 ) 
optimizerG = RMSprop(netG.parameters(),lr=0.00005 )  

for epoch in range(25):
    
    for i, data in enumerate(dataloader, 0):
        
        #1st Step: Updating the weights of the neural network of the discriminator
        for parm in netD.parameters():
                parm.data.clamp_(-0.01,0.01)
        netD.zero_grad()
        
        #Training the discriminator with a real image
        real, _ = data
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0]))
        output = netD(input)
        #output.backward()
        errD_real = torch.mean(output)
        
        #Training the discriminator with a generated image
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = netD(fake.detach())
        errD_fake = torch.mean(output)
        
        
        #Backpropogating the total error
        errD = errD_fake - errD_real
        errD.backward()
        #output2.backward()
        optimizerD.step()
        
        #2nd Step: Updating the weights of the nn of the generator
        
        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = netD(fake)
        #output.backward()
        errG = -torch.mean(output)
        errG.backward()
        optimizerG.step()
        
        #3rd Step: Priting the losses and saving real+generated images
        print('[%d/%d][%d/%d] ' % (epoch, 25, i, len(dataloader)))
        if ((i % 100 == 0)|(i==782)):
            print('save image for epoch:', epoch)
            vutils.save_image(real, '%s/real_samples.png' % ("./wgan_result/result1"), normalize = True)
            fake = netG(noise)
            vutils.save_image(fake.data, '%s/fake_samples.png' % ("./wgan_result/result2"), normalize = True)
            
            FID.fid("./wgan_result/result1","./wgan_result/result2")
                
        
