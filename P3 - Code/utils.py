import torch 
import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as copy

def iou(pred, target, n_class): #pass in a 3d tensor of non one-hot encoded vectors --> number of images,H,W
    dontInclude = set([0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]) #These are the classes with TrainId == 255
    #ious = []
    intermission = [] # intersection
    wasUpWithThat = [] #unions
    forCalculation = list()
    for cls in range(n_class):

        if cls in dontInclude:
            intermission.append(float('nan'))
            wasUpWithThat.append(float('nan'))
            continue 
         
        prediction = pred == cls
        targeted = target == cls
        
        intersection = (prediction & targeted).float().sum(dim =(1,2))
        union = (prediction | targeted).float().sum(dim = (1,2))
        
        wasUpWithThat.append(union.sum().item()) #for a particular class sum along the batch dimesnion
        intermission.append(intersection.sum().item()) #sum the intersection along the batch dimension     
           

    return torch.Tensor(intermission), torch.Tensor(wasUpWithThat) # torch.Tensor(ious), sum(forCalculation)/len(forCalculation)


def MeanIOU(combined):
    numLoops = 0
    runningSum = 0
    for el in combined:
        if math.isnan(el):
            continue
        numLoops += 1
        runningSum += el.item()
    return runningSum/numLoops


def pixel_acc(pred, target):
    #Create a boolean tensor of all the trainID != 255
    goodFlags = (target > 6) * (target != 29) * (target != 30)* (target != 18) * (target != 16) * (target != 15) * (target != 14) * (target != 10) * (target != 9) 
    
    #print("The sum of the non-negative entries is: " + str(goodFlags.sum().item()))
    
    #zeroOut the badFlags at the relevant indexes
    pred = pred*goodFlags
    target = target*goodFlags
    
    #print((pred != 0).sum().item())
    
    #get the sum to subtract by since the false indexes for both will now match (both will == 0)
    funWithFlags = torch.prod(torch.tensor(goodFlags.size()))
    howManyToSubtract = funWithFlags - goodFlags.sum() #dimensions - all the instances where the flags are good gives the sum of bad flags
    
    bringIt = torch.eq(pred, target) #Do the pixel predictions match? --> returns a T/F tensor of same dimensions as pred and target
    numerator = bringIt.sum().item() - howManyToSubtract.item()
    denom = torch.prod(torch.tensor(bringIt.size())).item() - howManyToSubtract.item()
    accuracy = 100*(numerator/denom) 
    return accuracy

def image_overlay(img, label, saving_dir, n_class):
    from matplotlib.cm import get_cmap
    fig,ax = plt.subplots(figsize=(6,5))
    cmap = get_cmap('viridis')
    
    # converts labels to some colour
    a = list(map(lambda x: cmap((x / n_class * cmap.N).astype(int)), label))
    
    overlay = 0.7*np.array(a)[:, :,0:3] + 0.3*img
    plt.imshow(overlay)
    plt.imsave(saving_dir, overlay)
    
def data2img(torch_img):
    img = torch_img.permute(1, 2, 0).numpy()
    for i in range(3):
        img[:, :, i] -= np.min(img[:, :, i])
        img[:, :, i] /= np.max(img[:, :, i])
    return img    

class EarlyStop:
    def __init__(self, patience):
        self.patience = patience
        self.virtue = 0
        self.prevLoss = float('inf')
        self.readySetStop = False
        self.theOneModelToRuleThemAll = None 
    
    def __call__(self, valLoss, bottomOfTheBarrel):
        if self.readySetStop:
            print("Why are you still training???  You are overfitting...")
        elif (valLoss < self.prevLoss):
            self.virtue = 0
            self.prevLoss = valLoss
            self.theOneModelToRuleThemAll = copy(bottomOfTheBarrel)
            print("Going down no need to do anything... good job!")
        else:
            print("Entering the thunderdome...")
            self.prevLoss = valLoss
            self.virtue += 1 
            if self.virtue >= self.patience:
                self.readySetStop = True
                print("You have won the hunger games")
               
        return self.readySetStop, self.theOneModelToRuleThemAll
    
def weighted_cross_entropy(output, labels, weights):
    batch_size = output.size(0)
    num_classes = output.size(1)
    H = output.size(2)
    W = output.size(3)
    
    # Normalize the weights
    weights = num_classes * weights/weights.sum()
    
    # Take the log softmax of the output
    logp = -1.0 * torch.nn.functional.log_softmax(output, dim = 1)
    
    # Gather along the correct classes and multiply
    logp.gather(1, labels.view(batch_size, 1, H, W))
   
    # Multiply by the weights
    weights = weights.view(1, num_classes, 1, 1)
    weighted_logp = (logp * weights)
    
    # Normalize the weights
    weighted_loss = weighted_logp.sum(1)/weights.sum(1)
    return weighted_loss.mean()