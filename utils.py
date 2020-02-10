import torch 
#import math 

def iou(pred, target, n_class): #pass in a 3d tensor of non one-hot encoded vectors --> number of images,H,W
    dontInclude = set([0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]) #These are the classes with TrainId == 255
    ious = []
    forCalculation = list()
    for cls in range(n_class):

        if cls in dontInclude:
            ious.append(float('nan'))
            continue 
         
        prediction = pred == cls
        targeted = target == cls
        
        intersection = (prediction & targeted).float().sum(dim =(1,2))
        union = (prediction | targeted).float().sum(dim = (1,2))
        
        #if cls == 12:
        #    print(intersection)
        #    print(union)
        #smoothCriminal = 0.000001
        
        if min(union).item() == 0: 
            #some images don't have any pizels while some do - this will need to be fixed in the future once get metric clarity
            
            #print("This shouldn't be happening...." + str(cls)) 
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            
            toAppend = (intersection/union) #(union+smoothCriminal)) #handle 0/0
            
            
            #if math.isnan(toAppend):
            #    print("This makes no sense..." + str(cls))
            #    print(intersection)
            #    print(union)
            #print("Taking the sum:")
            #print(sum(toAppend)
            #print("getting the len:")
            #print(len(toAppend))
            
            toAppend = sum(toAppend)/len(toAppend)
            
            #if math.isnan(toAppend):
            #    print("This makes no sense..." + str(cls))
            #    print(intersection)
            #    print(union)
            #    print(intersection/union)
            
            ious.append(toAppend.item())
            forCalculation.append(toAppend.item())
            
    #print("doingThings!!!")
    
    #print(forCalculation)
    
    #return the ious of all classes and the mean iou of the batch; the former is done to return an indexed array to match 
    #with specific classes IOU that can be accessed as requested in the assignment (e.g. building: 11)
    
    return torch.Tensor(ious), sum(forCalculation)/len(forCalculation)


def pixel_acc(pred, target):
    bringIt = torch.eq(pred, target) #Do the pixel predictions match? --> returns a T/F tensor of same dimensions as pred and target
    accuracy = 100* (bringIt.sum().item()/torch.prod(torch.tensor(bringIt.size())).item()) 
    return accuracy
