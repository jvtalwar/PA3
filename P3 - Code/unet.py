"""
Created on Wed Feb 12 14:19:27 2020

@author: Bojing

U-Net v1.0
"""

import torch
import torch.nn as nn

# kernel, stride, padding set as default per UNet paper
def double_conv_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
   
    ''' conv -> batch norm -> ReLU -> conv -> batch norm -> ReLU '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True),
        nn.Dropout(0.5)
        #nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
        #nn.BatchNorm2d(out_channels),
        #nn.ReLU(inplace = True)            
    )

# don't wrap maxpool into double_conv_bn block, to save intermediate priors
def down_maxpool():
   
    return nn.MaxPool2d(2)

# upsampling kernel, stride, set as default per UNet paper
# no need for padding (dim red)
def up_unpool_test(in_channels, out_channels, kernel_size=2, stride=2):
   
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True)
    )

class unet(nn.Module):
   
    # Per UNet Paper: 64 -> 128 -> 256 -> 512 -> 1024, then upsample
   
    def __init__(self, in_channels, n_class):
        super().__init__()
        self.n_class = n_class
       
        # Downsample (CONTRACTING)
        self.conv1 = double_conv_bn(in_channels, 64)
        self.conv2 = double_conv_bn(64, 128)
        self.conv3 = double_conv_bn(128, 256)
        self.conv4 = double_conv_bn(256, 512)
        self.conv5 = double_conv_bn(512, 1024)
       
        # call later on each downsampled layer
        self.down_maxpool = down_maxpool()
       
        # Upsample (EXPANSIVE)
        self.up_unpool1 = up_unpool_test(1024, 512)
        self.conv6 = double_conv_bn(1024, 512)  # NB: concatenate after this step to remake 512 + 512 -> 1024
       
        self.up_unpool2 = up_unpool_test(512, 256)
        self.conv7 = double_conv_bn(512, 256)
       
        self.up_unpool3 = up_unpool_test(256, 128)
        self.conv8 = double_conv_bn(256, 128)
       
        self.up_unpool4 = up_unpool_test(128, 64)
        self.conv9 = double_conv_bn(128, 64)
       
        # generate output layer
        # NB: Just as no maxpooling in bottom, no upsamling in top (output)
        self.output = nn.Conv2d(64, n_class, 1)
       
    def forward(self, x):
       
        #####
        x1 = self.conv1(x) # later used as prior (skip cxn) for up_x4
        p_x1 = self.down_maxpool(x1)
       
        x2 = self.conv2(p_x1) # later used as prior (skip cxn) for up_x3
        p_x2 = self.down_maxpool(x2)
       
        x3 = self.conv3(p_x2) # later used as prior (skip cxn) for up_x2
        p_x3 = self.down_maxpool(x3)
       
        x4 = self.conv4(p_x3) # later used as prior (skip cxn) for up_x1
        p_x4 = self.down_maxpool(x4)
       
        x5 = self.conv5(p_x4) # reached bottom
        
        #####
        up_x1 = self.up_unpool1(x5)
        x6 = torch.cat([up_x1, x4], dim=1)
        x6 = self.conv6(x6)
       
        up_x2 = self.up_unpool2(x6)
        x7 = torch.cat([up_x2, x3], dim=1)
        x7 = self.conv7(x7)
       
        up_x3 = self.up_unpool3(x7)
        x8 = torch.cat([up_x3, x2], dim=1)
        x8 = self.conv8(x8)
       
        up_x4 = self.up_unpool4(x8)
        x9 = torch.cat([up_x4, x1], dim=1)
        print(x9.size())
        x9 = self.conv9(x9)
       
        output = self.output(x9) # reached top
        print(output.size())
        return output
