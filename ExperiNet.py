# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:03:38 2020

@author: james
"""

import torch.nn as nn
import torch.nn.functional as F

class ExperiNet(nn.Module):
    
    def __init__(self, n_class):
        super(ExperiNet,self).__init__()
        self.n_class = n_class
        inputOutput = [64,128,256,512,512] # the number of channels per layer
        self.enigma = nn.ModuleList()
        self.christopher = nn.ModuleList()
        
        encoderParameters = [3] + inputOutput #need the input channels...
        print(encoderParameters)
        
        #decoder --> need to reverse the layers 
        decoderParameters = inputOutput[::-1] + [inputOutput[0]]
        print(decoderParameters)
        
        #depth of each layer
        layerDepth = [2,2,2,2,3]
        backwards = [3,2,2,2,1] 
        self.layerDepth = layerDepth
        
        for i in range(len(layerDepth)):
            print(len(encoderParameters))
            self.enigma.append(Encoder(encoderParameters[i], encoderParameters[i+1], layerDepth[i]))
            self.christopher.append(Decoder(decoderParameters[i], decoderParameters[i+1], backwards[i]))
                
        self.classifier = nn.Conv2d(inputOutput[0], self.n_class, kernel_size = 3, padding = 1, stride = 1, dilation = 1)
            
    def forward (self, x):
        whereAmIMappingTo = list()
        howBigDoIWantToBeWhenIGrowUp = list()
        #Encode
        for i in range(len(self.layerDepth)):
            (x,index),bigMac = self.enigma[i](x)
            whereAmIMappingTo.append(index)
            howBigDoIWantToBeWhenIGrowUp.append(bigMac)
        
        #Decode
        #print("Sizes...:")
        #print(howBigDoIWantToBeWhenIGrowUp)
              
        for i in range(len(self.layerDepth)):
            #print("entering...:")
            #print(i)
            x = self.christopher[i](x, whereAmIMappingTo[len(whereAmIMappingTo) - 1 - i], howBigDoIWantToBeWhenIGrowUp[len(howBigDoIWantToBeWhenIGrowUp) -1-i])
            
        x = self.classifier(x) #get the score
        return x

class Encoder(nn.Module):
    def __init__(self, into,outo, howManyLayers):
        super(Encoder,self).__init__()
        
        villainLayer = [nn.Conv2d(into, outo, kernel_size = 3, padding = 1, stride = 1),
                       nn.BatchNorm2d(outo),
                       nn.ReLU(inplace=True)]
        
        for i in range(howManyLayers-1):
            villainLayer += [nn.Conv2d(outo, outo, kernel_size = 3, padding = 1, stride = 1),nn.BatchNorm2d(outo),               nn.ReLU(inplace=True)]
        print("vill layer num : " + str(len(villainLayer)))
        
     
    
        self.comprehensive = nn.Sequential(*villainLayer)  #unzip from list form
        
    def forward(self,x):
        #need the maxpool indices abd suze for unpooling in the decoder structure
        preMaxPooling = self.comprehensive(x)
        
        #applyMax pooling with 2x2 kernel and 2 stride
        
        return F.max_pool2d(preMaxPooling, 2, 2, return_indices  = True), preMaxPooling.size() 
    
class Decoder(nn.Module):
    
    def __init__(self, into, outo, howManyLayers):
        super(Decoder, self).__init__()
        
        breakEnigma = list()
        for i in range(howManyLayers-1):
            breakEnigma = [nn.Conv2d(into, into, kernel_size = 3, padding = 1, stride = 1), nn.BatchNorm2d(into), nn.ReLU(inplace = True)]
        
       
        
        breakEnigma += [nn.Conv2d(into, outo, kernel_size = 3, padding = 1, stride =1), nn.BatchNorm2d(outo), nn.ReLU(inplace = True)]
            
            
        self.downingStreet = nn.Sequential(*breakEnigma)
        
    def forward(self,x, indxs, size):
        wellNeedUnpooling = F.max_unpool2d(x, indxs,kernel_size = 2, stride = 2, padding = 0, output_size = size)
        
        #Apply Maxpooling then do everything
        
        return self.downingStreet(wellNeedUnpooling)
        
