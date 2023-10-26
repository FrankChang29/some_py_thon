
import torch
import torch.nn as nn

if __name__=="__main__":
    im = nn.Linear(5,10)
    t = torch.randn(4,5)
    print(im(t).shape)
    print(im.named_parameters())