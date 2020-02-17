CSE 253: Programming Assignment 3

In this submission folder, the following files relating to this assignment are included: 

1) final_model_train.ipynb/py -- General script for training and testing a given model. See below for the instructions on how to      change parameters to produce desired outputs for each model
2) basic_fcn.py -- baseline model implementation in PyTorch
3) ExperiNet.py -- experimental model implementation in PyTorch
4) TransResNet.py -- transfer learning model implementation in PyTorch using ResNet18
5) unet.py -- our U-Net implementation in PyTorch
6) dataloader.py -- updated dataloader file for loading data adding in transformations
7) utils.py -- all utility functions used in training (includes accuracy, IoU, early stopping etc.)
8) check_class_balance.ipynb -- script to check the class imbalance in the training set
9) total_pixel_weights.npy -- numpy array object to be loaded in for weighted training
10) train.csv, val.csv, test.csv -- paths to train, validation and test images on UCSD GPU server respectively

Running the above scripts requires one to be logged on to the UCSD GPU server where the data is housed. 

Instructions For Running: 

To generate outputs similar to those in our report, the final_model_train.ipynb/py scripts will need to be run with certain parameters depending on the desired output. We list below the parameters for each part of the report. The script will output all the necessary plots and images to the current working directory following training and testing. Currently the script is set up to run A.

A. Baseline model, unweighted, no augmentation:
    1. Change model name after import statements to desired output file name
    2. Make sure you specify 50 epochs with patience set to 5
    3. Make sure that you are loading the appropriate model implementation in line 293 and line 275 of the .py file (i.e. an              object of class 'FCN' should be instantiated at these two lines)
    
B. Basline model. weighted cross entropy, no augmentation
    1. All the same as in A, except in lines 303, 304 and 306, change 'criterion' to 'weighted_criterion'
    
C. Baseline model, unweighted, augmentation:
    1. All the same as in A, except in line 26, change transforms passed into 'train_dataset' from 'None' to 
       '['hflip',  'rotation']
       
D. ExperiNet:
    1. Change model name after import statements to desired output file name
    2. Change training batch size to 3 and validation batch size to 1 at line 24
    3. Change transforms passed into 'train_dataset' from 'None' to 
       '['hflip',  'rotation']
    4. Make sure you specify 50 epochs with patience set to 3
    5. Make sure that you are loading the appropriate model implementation in line 293 and line 275 of the .py file (i.e. an          object of class 'ExperiNet' should be instantiated at these two lines)
    
E. Transfer ResNet:
    1. Change model name after import statements to desired output file name
    3. Change transforms passed into 'train_dataset' from 'None' to 
       '['hflip',  'rotation']
    4. Make sure you specify 50 epochs with patience set to 5
    5. Make sure that you are loading the appropriate model implementation in line 293 and line 275 of the .py file (i.e. an          object of class 'TransResNet' should be instantiated at these two lines)
    
F. UNet:
    1. Change model name after import statements to desired output file name
    2. Change validation batch size to 1 at line 24
    3. Change transforms passed into 'train_dataset' from 'None' to 
       '['hflip',  'rotation']
    4. Make sure you specify 50 epochs with patience set to 5
    5. Make sure that you are loading the appropriate model implementation in line 293 and line 275 of the .py file (i.e. an          object of class 'unet' should be instantiated at these two lines)
