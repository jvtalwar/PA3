import torch 
import math 

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
