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
    bringIt = torch.eq(pred, target) #Do the pixel predictions match? --> returns a T/F tensor of same dimensions as pred and target
    accuracy = 100* (bringIt.sum().item()/torch.prod(torch.tensor(bringIt.size())).item()) 
    return accuracy
