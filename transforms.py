import numpy as np

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        i=sample["image"]
        l=sample["labels"]
        
        h, w = i.shape[1:3] #get height and width of image
        new_h, new_w = self.output_size #get new height and width
        
        top = np.random.randint(0, h - new_h) #pick random height start
        left = np.random.randint(0, w - new_w) #pick random width start

        i = i[:,top: top + new_h,
                      left: left + new_w]

        l = l[top: top + new_h,
              left: left + new_w]
        
        return i, l