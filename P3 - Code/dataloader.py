from torch.utils.data import Dataset, DataLoader# For custom data-sets
import torchvision.transforms as pt_transforms
import torchvision.transforms.functional as TF
#import torchsample
#from torchsample.transforms import RandomRotate
from skimage.transform import rotate as Rotate
import numpy as np
from PIL import Image
import torch
import pandas as pd
from collections import namedtuple
import random

n_class    = 34
means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).

    'trainId'     , # An integer ID that overwrites the ID above, when creating ground truth
                    # images for training.
                    # For training, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

labels_classes = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'ground'          , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'ground'          , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'ground'          , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'ground'          , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) )
]

max_translate_x, max_translate_y = 0.2, 0.2
max_rotation = 10

class CityScapesDataset(Dataset):

    def __init__(self, csv_file, n_class=n_class, transforms=None, resize_factor = 1):
        """ 
        Transforms is a list that include the list of transformations to be applied
        """
        
        self.data      = pd.read_csv(csv_file)
        self.means     = means
        self.n_class   = n_class
        self.data_size = np.asarray(Image.open(self.data.iloc[0, 0]).convert('RGB')).shape[:2]
        self.data_size = tuple([np.int(s * resize_factor) for s in self.data_size])
        #(np.int(self.data_size[0]*resize_factor),np.int(self.data_size[0]*resize_factor/2))   
        
        self.resize_factor = resize_factor
        
        # Add any transformations here
        trans_list = []
        if transforms is not None:
            for trans in transforms:
                if 'translation' == trans:#in transforms:
                    trans_list.append(pt_transforms.RandomAffine(degrees = 0, translate = (max_translate_x, max_translate_y)))
                if 'rotation' ==trans: #in transforms:   
                    trans_list.append(pt_transforms.RandomRotation(degrees = max_rotation))
                if 'hflip' == trans: # in transforms:
                    trans_list.append(pt_transforms.RandomHorizontalFlip(p=1))
                if 'crop' == trans: #in transforms:
                #print("hello")
                    trans_list.append(pt_transforms.RandomCrop(512, padding = 0)) #(self.data_size, scale=(0.5, 1), ratio = (1.8, 2.2)))
                #print(trans_list)
            #self.transforms = None
            self.transforms = CustomCompose(trans_list)
        else:
            self.transforms = None
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name   = self.data.iloc[idx, 0]

        img_init = Image.open(img_name)
        label_name = self.data.iloc[idx, 1]
        label_init = Image.open(label_name)

        # make a copy of the initial images
        img = img_init.copy()
        label = label_init.copy()
        
        # resize the data if requested
        if self.resize_factor != 1:
            img = pt_transforms.functional.resize(img, self.data_size)
            label = pt_transforms.functional.resize(label, self.data_size, interpolation = Image.NEAREST)
            
        # applying the transformation on the copy
        if self.transforms is not None:
            #print(self.transforms)
            img, label = self.transforms(img, label)
        
        if img.size[0] != self.data_size[0]:
            img = pt_transforms.functional.resize(img, self.data_size)
            label = pt_transforms.functional.resize(label, self.data_size, interpolation = Image.NEAREST)
        # converting to numpy arrays
        img = np.asarray(img.convert('RGB'))
        label      = np.asarray(label)
        img_init = np.asarray(img_init.convert('RGB'))
        label_init= np.asarray(label_init)

        # reduce mean
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]
         
        
        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        # create one-hot encoding
        h, w = label.shape
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1
        #print("target_size is: " + str(target.size()))
        return img,target, label
    
class CustomCompose(pt_transforms.Compose):
    """ This class extends transforms.Compose and contains a list of
        transformations which are applied both image and label.
    """
    def __init__(self, trans_list):
        super(CustomCompose, self).__init__(trans_list)
        self.transforms = trans_list
        self.comeAtMeBro = 200 #what the reduced dimesions should be --> assuming a square 
        
    def __call__(self, img, label):
        for t in self.transforms:
            # for affine-derived transformations
            if isinstance(t, pt_transforms.RandomAffine):
                params = t.get_params(t.degrees, t.translate, t.scale, t.shear, img.size)
                img = TF.affine(img, *params) #, resample=t.resample, fillcolor=t.fillcolor)
                label = TF.affine(label, *params) #, resample=False, fillcolor=False)
            if isinstance(t, pt_transforms.RandomRotation):
                #img = Rotate(img, angle = max_rotation, mode = 'wrap')
                #label = Rotate(label, angle = max_rotation, mode = 'wrap')
                if random.random() < 0.3:
                    #print("rotating...")
                    img = TF.rotate(img, max_rotation) #, resample=t.resample, fillcolor=t.fillcolor)
                    label = TF.rotate(label, max_rotation) #, resample=False, fillcolor=False)
                #print(params)
            if isinstance(t, pt_transforms.RandomHorizontalFlip):
                if random.random() < 0.5:
                    #print("flipping")
                    img = TF.hflip(img)
                    label = TF.hflip(label)
            if isinstance(t, pt_transforms.RandomCrop):
                if random.random() < 0.23:
                    #print("cropping...")
                    #print("hello?")
                    #print("The t is:" + str(t))
                    #print("The paramaters are: " + str(t.get_params(img)) )
                    cropIndexes = t.get_params(img,output_size=(self.comeAtMeBro, self.comeAtMeBro*2))
                    i, j, h, w = cropIndexes
                    img = TF.crop(img, i, j, h, w)
                    #img = pt_transforms.functional.resize(img, (self.comeAtMeBro*4, self.comeAtMeBro*8))
                    
                    label = TF.crop(label, i,j,h,w)
                    #label = pt_transforms.functional.resize(img, (self.comeAtMeBro*4, self.comeAtMeBro*8))
                    #img = t.(img)
                    #label = t.(label)
                    #i, j, h, w = t.get_params(img) #, t.scale, t.ratio)
                    #print([i, j, h, w, t.scale, t.ratio])
                    #img = TF.resized_crop(img, i, j) #, h, w, t.size, t.interpolation)
                    #label = TF.resized_crop(label, i, j, h, w, t.size, Image.NEAREST)
        return img, label
