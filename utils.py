import torch 

def iou(pred, target):
    ious = []
    for cls in range(n_class):
        # Complete this function
        #intersection = # intersection calculation
        # union = #Union calculation
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            print("hi")

            # Append the calculated IoU to the list ious
    return ious


def pixel_acc(pred, target):
    bringIt = torch.eq(pred, target) #Do the pixel predictions match? --> returns a T/F tensor of same dimensions as pred and target
    accuracy = 100* (bringIt.sum().item()/torch.prod(torch.tensor(bringIt.size())).item()) 
    return accuracy