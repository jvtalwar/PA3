import torch.nn as nn
import torchvision
from collections import OrderedDict


class TransResNet(nn.Module):

    def __init__(self, n_class, base_model = 'resnet18'):
        super().__init__()
        self.n_class = n_class
        
        # defining basic model by ablating the last two layers (average and linear) from ResNet
        if base_model == 'resnet18':
            resnet = torchvision.models.resnet18(pretrained = True)
        elif base_model == 'resnet34':
            resnet = torchvision.models.resnet34(pretrained=True)

        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        del resnet
        
        # freezing the learning for ResNet
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.relu    = nn.ReLU(inplace=True)
        
        # output of the convolutional encoder in resnet is 512 and compatible with our decoder
        self.decoder = nn.Sequential(OrderedDict([('deconv1', self.deconv_internal(512, 512)),
                                    ('deconv2', self.deconv_internal(512, 256)),
                                    ('deconv3', self.deconv_internal(256, 128)),
                                    ('deconv4', self.deconv_internal(128, 64)),
                                    ('deconv5', self.deconv_internal(64, 32)),
                                    ('output', nn.Conv2d(32,n_class, kernel_size=1))]))
        
#         self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn1     = nn.BatchNorm2d(512)
#         self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn2     = nn.BatchNorm2d(256)
#         self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn3     = nn.BatchNorm2d(128)
#         self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn4     = nn.BatchNorm2d(64)
#         self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn5     = nn.BatchNorm2d(32)
#         self.classifier = nn.Conv2d(32,n_class, kernel_size=1)
    
    def deconv_internal(self, in_channels, out_channels):
        """ Layers consisting of one deconvolution layer and one ReLU """
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace = True))
    
    def initialize_weights(self, weight_func):
        """For initializing the decoder weights, without changing the encoder weights"""
        self.decoder.apply(weight_func)
        return 
    
    def forward(self, x):
        # encoder 
        x = self.encoder(x)
        # A ReLU after encoding with resnet
        x = self.relu(x)
        
        # the rest of the decoder
        x = self.decoder(x)
        
        return x  # size=(N, n_class, x.H/1, x.W/1)