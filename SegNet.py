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
        #self.conv3Three = nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1)
        self.conv4One = nn.Conv2d(256, 512, kernel_size=3, padding=1, dilation=1)
        self.conv4Two = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        #self.conv4Three = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        #self.conv5One = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        #self.conv5Two = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        #self.conv5Three = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        
        #self.deconv5One = nn.ConvTranspose(512, 512, kernel_size=3, padding=1, dilation=1)
        #self.deconv5Two = nn.ConvTranspose(512, 512, kernel_size=3, padding=1, dilation=1)
        #self.deconv5Three = nn.ConvTranspose(512, 512, kernel_size=3, padding=1, dilation=1)
        self.deconv4One = nn.Conv2d(512, 256, kernel_size=3, padding=1, dilation=1)
        self.deconv4Two = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        #self.deconv4Three = nn.ConvTranspose(512, 512, kernel_size=3, padding=1, dilation=1)
        self.deconv3One = nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1)
        self.deconv3Two = nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1)
        #self.deconv3Three = nn.ConvTranspose(256, 256, kernel_size=3, padding=1, dilation=1)
        self.deconv2One = nn.Conv2d(128, 64, kernel_size=3, padding=1, dilation=1)
        self.deconv2Two = nn.Conv2d(128, 128, kernel_size=3, padding=1, dilation=1)
        self.deconv1Two = nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=1)
        self.deconv1One = nn.Conv2d(64, n_class, kernel_size=1)
        
        #batchnorm
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4and5 = nn.BatchNorm2d(512)
        
        #activation
        self.relu = nn.ReLU(inplace=True)
        
        #poooling and unpooling
        self.mp1 = nn.MaxPool2d(2,2, return_indices = True)
        self.mp2 = nn.MaxPool2d(2,2, return_indices = True)
        self.mp3 = nn.MaxPool2d(2,2, return_indices = True)
        self.mp4 = nn.MaxPool2d(2,2, return_indices = True)
        self.mp5 = nn.MaxPool2d(2,2, return_indices = True)
        self.unpool1 = nn.MaxUnpool2d(2,2)
        self.unpool2 = nn.MaxUnpool2d(2,2)
        self.unpool3 = nn.MaxUnpool2d(2,2)
        self.unpool4 = nn.MaxUnpool2d(2,2)
        #self.unpool5 = nn.MaxUnpool2d(2,2)
        
    def forward(self, x):
        
        #need the size and indexes at each layer for the unpooling operation
        #ENCODER:
        x = self.relu(self.bn1(self.conv1One(x)))
        x = self.relu(self.bn1(self.conv1Two(x)))
        uno = x.size()
        x, firstIndexes = self.mp1(x)
        
        
        x = self.relu(self.bn2(self.conv2One(x)))
        x = self.relu(self.bn2(self.conv2Two(x)))
        dos = x.size()
        x, secondIndexes = self.mp2(x)
        
        x = self.relu(self.bn3(self.conv3One(x)))
        x = self.relu(self.bn3(self.conv3Two(x)))
        #x = self.relu(self.bn3(self.conv3Three(x)))
        tres = x.size()
        x, thirdIndexes = self.mp3(x)
        
        x = self.relu(self.bn4and5(self.conv4One(x)))
        x = self.relu(self.bn4and5(self.conv4Two(x)))
        #x = self.relu(self.bn4and5(self.conv4Three(x)))
        cuatro = x.size()
        x, fourthIndexes = self.mp4(x)
        
        #x = self.relu(self.bn4and5(self.conv5One(x)))
        #x = self.relu(self.bn4and5(self.conv5Two(x)))
        #x = self.relu(self.bn4and5(self.conv5Three(x)))
        #cinco = x.size()
        #x, fifthIndexes = self.mp5(x)
        
        #DECODER:
        
        #x = self.unpool5(x, fifthIndexes, output_size = cinco)
        #x = self.relu(self.bn4and5(self.deconv5Three(x)))
        #x = self.relu(self.bn4and5(self.deconv5Two(x)))
        #x = self.relu(self.bn4and5(self.deconv5One(x)))
        
        x = self.unpool4(x, fourthIndexes, output_size = cuatro)
        #x = self.relu(self.bn4and5(self.deconv4Three(x)))
        x = self.relu(self.bn4and5(self.deconv4Two(x)))
        x = self.relu(self.bn3(self.deconv4One(x)))
        
        x = self.unpool3(x, thirdIndexes, output_size = tres)
        #x = self.relu(self.bn3(self.deconv3Three(x)))
        x = self.relu(self.bn3(self.deconv3Two(x)))
        x = self.relu(self.bn2(self.deconv3One(x)))
        
        x = self.unpool2(x, secondIndexes, output_size = dos)
        x = self.relu(self.bn2(self.deconv2Two(x)))
        x = self.relu(self.bn1(self.deconv2One(x)))
        
        x = self.unpool1(x, firstIndexes, output_size = uno)
        x = self.relu(self.bn1(self.deconv1Two(x)))
        x = self.deconv1One(x)
        
        return x
        
        
        
        
        
        
