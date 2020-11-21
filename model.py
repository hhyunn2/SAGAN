import torch
import torch.nn as nn

from attention_module import Atten_Module

def conv2d(params_list, batchnorm=True):
    channel_in, channel_out, kernel_size, stride, padding, activation = params_list
    layers = []
    if batchnorm:
        layers += [nn.utils.spectral_norm(nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding, bias=False)), nn.BatchNorm2d(channel_out)]
        nn.init.xavier_uniform_(layers[0].weight)
    else:
        layers += [nn.utils.spectral_norm(nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding, bias=False))]
        nn.init.xavier_uniform_(layers[0].weight)
        
    if activation == 'ReLU':
        layers += [nn.ReLU(inplace=True)]
    if activation == 'LeakyReLU':
        layers += [nn.LeakyReLU(0.2, inplace=True)]
    if activation == 'Tanh':
        layers += [nn.Tanh()]
    if activation == 'Sigmoid':
        layers += [nn.Sigmoid()]
        
    return nn.Sequential(*layers)
        
    
def upconv2d(params_list, batchnorm = True):
    channel_in, channel_out, kernel_size, stride, padding, activation = params_list
    layers = []
    if batchnorm:
        layers += [nn.utils.spectral_norm(nn.ConvTranspose2d(channel_in, channel_out, kernel_size, stride, padding, bias=False)), nn.BatchNorm2d(channel_out)]
        nn.init.xavier_uniform_(layers[0].weight)
    else:
        layers += [nn.utils.spectral_norm(nn.ConvTranspose2d(channel_in, channel_out, kernel_size, stride, padding, bias=False))]
        nn.init.xavier_uniform_(layers[0].weight)
        
    if activation == 'ReLU':
        layers += [nn.ReLU(inplace=True)]
    if activation == 'LeakyReLU':
        layers += [nn.LeakyReLU(0.2, inplace=True)]
    if activation == 'Tanh':
        layers += [nn.Tanh()]
    if activation == 'Sigmoid':
        layers += [nn.Sigmoid()]
        
    return nn.Sequential(*layers)
    
    
cfg_g = [[100, 1024, 4, 1, 0, 'ReLU'], [1024, 512, 4, 2, 1, 'ReLU'], [512, 256, 4, 2, 1, 'ReLU'], [256, 128, 4, 2, 1, 'ReLU'], [128, 64, 4, 2, 1, 'ReLU'],[64, 3, 4, 2, 1, 'Tanh']]


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
       

        self.upconv1 = upconv2d(cfg_g[0])
        self.upconv2 = upconv2d(cfg_g[1])
        self.upconv3 = upconv2d(cfg_g[2])
        self.upconv4 = upconv2d(cfg_g[3])
        self.upconv5 = upconv2d(cfg_g[4])
        self.upconv6 = upconv2d(cfg_g[5], batchnorm = False)
        
    def forward(self, x):
        out = self.upconv1(x)
        out = self.upconv2(out)
 
        out = self.upconv3(out)
        out = self.upconv4(out)
        
        out = self.upconv5(out)
        out = self.upconv6(out)
    
        return out
      
      
cfg_d = [[3, 64, 4, 2, 1, 'LeakyReLU'], [64, 128, 4, 2, 1, 'LeakyReLU'], [128, 256, 4, 2, 1, 'LeakyReLU'], [256, 512, 4, 2, 1, 'LeakyReLU'], [512, 1024, 4, 2, 1, 'LeakyReLU'], [1024, 1, 4, 1, 0, 'Sigmoid']]


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
       
        self.conv1 = conv2d(cfg_d[0])
        self.conv2 = conv2d(cfg_d[1])
        self.conv3 = conv2d(cfg_d[2])
        self.conv4 = conv2d(cfg_d[3])
        self.conv5 = conv2d(cfg_d[4])
        self.conv6 = conv2d(cfg_d[5], batchnorm = False)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
 
        out = self.conv3(out)
        out = self.conv4(out)
    
        out = self.conv5(out)
        out = self.conv6(out)

        return out
