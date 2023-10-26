import torch
import torch.nn as nn

class LinearModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.W = nn.Parameter(torch.randn(dim,1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self,X):
        return X.mm(self.W)+self.b

if __name__=="__main__":
    tmpModule=LinearModule(5)
    x=torch.randn(4,5,requires_grad=True)
    m=tmpModule.forward(x)
    m.backward()
    print(x.grad)
    print(list(tmpModule.named_parameters()))