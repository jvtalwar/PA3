# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 21:13:35 2020

@author: james talwar

My attempt to implement SegNet for semantic segmentation PA3

"""

import torch.nn as nn

class SegNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        #Convolutions
        self.conv1One = nn.Conv2d(3, 64, kernel_size=3, padding=1, dilation=1)
        self.conv1Two = nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=1)
        self.conv2One = nn.Conv2d(64, 128, kernel_size=3, padding=1, dilation=1)
        self.conv2Two = nn.Conv2d(128, 128, kernel_size=3, padding=1, dilation=1)
        self.conv3One = nn.Conv2d(128, 256, kernel_size=3, padding=1, dilation=1)
        self.conv3Two = nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1)
        self.conv3Three = nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1)
        self.conv4One = nn.Conv2d(256, 512, kernel_size=3, padding=1, dilation=1)
        self.conv4Two = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv4Three = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5One = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5Two = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5Three = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        
        self.deconv5One = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.deconv5Two = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.deconv5Three = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.deconv4One = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1, dilation=1)
        self.deconv4Two = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.deconv4Three = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.deconv3One = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, dilation=1)
        self.deconv3Two = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1, dilation=1)
        self.deconv3Three = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1, dilation=1)
        self.deconv2One = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, dilation=1)
        self.deconv2Two = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, dilation=1)
        self.deconv1Two = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, dilation=1)
        self.deconv1One = nn.ConvTranspose2d(64, n_class, kernel_size=3, padding=1, dilation=1)
        
        #batchnorm
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4and5 = nn.BatchNorm2d(512)
        
        #activation
        self.relu = nn.ReLU(inplace=True)
        
        #poooling and unpooling
        self.mp = nn.MaxPool2d(2,2, return_indices = True)
        self.unpool = nn.MaxUnpool2d(2,2)
        
    def forward(self, x):
        
        #need the size and indexes at each layer for the unpooling operation
        #ENCODER:
        x = self.relu(self.bn1(self.conv1One(x)))
        x = self.relu(self.bn1(self.conv1Two(x)))
        uno = x.size()
        x, firstIndexes = self.mp(x)
        
        
        x = self.relu(self.bn2(self.conv2One(x)))
        x = self.relu(self.bn2(self.conv2Two(x)))
        dos = x.size()
        x, secondIndexes = self.mp(x)
        
        x = self.relu(self.bn3(self.conv3One(x)))
        x = self.relu(self.bn3(self.conv3Two(x)))
        x = self.relu(self.bn3(self.conv3Three(x)))
        tres = x.size()
        x, thirdIndexes = self.mp(x)
        
        x = self.relu(self.bn4and5(self.conv4One(x)))
        x = self.relu(self.bn4and5(self.conv4Two(x)))
        x = self.relu(self.bn4and5(self.conv4Three(x)))
        cuatro = x.size()
        x, fourthIndexes = self.mp(x)
        
        x = self.relu(self.bn4and5(self.conv5One(x)))
        x = self.relu(self.bn4and5(self.conv5Two(x)))
        x = self.relu(self.bn4and5(self.conv5Three(x)))
        cinco = x.size()
        x, fifthIndexes = self.mp(x)
        
        #DECODER:
        x = self.unpool(x, fifthIndexes, output_size = cinco)
        x = self.relu(self.bn4and5(self.deconv5Three(x)))
        x = self.relu(self.bn4and5(self.deconv5Two(x)))
        x = self.relu(self.bn4and5(self.deconv5One(x)))
        
        x = self.unpool(x, fourthIndexes, output_size = cuatro)
        x = self.relu(self.bn4and5(self.deconv4Three(x)))
        x = self.relu(self.bn4and5(self.deconv4Two(x)))
        x = self.relu(self.bn3(self.deconv4One(x)))
        
        x = self.unpool(x, thirdIndexes, output_size = tres)
        x = self.relu(self.bn3(self.deconv3Three(x)))
        x = self.relu(self.bn3(self.deconv3Two(x)))
        x = self.relu(self.bn2(self.deconv3One(x)))
        
        x = self.unpool(x, secondIndexes, output_size = dos)
        x = self.relu(self.bn2(self.deconv2Two(x)))
        x = self.relu(self.bn1(self.deconv2One(x)))
        
        x = self.unpool(x, firstIndexes, output_size = uno)
        x = self.relu(self.bn1(self.deconv1Two(x)))
        x = self.deconv1One(x)
        
        return x
        
        
        
