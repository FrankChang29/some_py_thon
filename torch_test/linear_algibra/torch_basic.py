import torch

if __name__ == "__main__":
    a = torch.normal(torch.Tensor([1.]),torch.Tensor([3.]))
    print(a)
    a = torch.normal(torch.arange(1.,10.,2),torch.arange(1,0.8,-1)).reshape(5,1)
    print(torch.var(a))
    a = torch.arange(24).reshape(2,3,1,1,4)
    print(torch.squeeze(a))
    c=torch.unsqueeze(a,5)
    print(c.shape)

    ori1 = torch.rand(1,3,2)
    or2 = torch.rand(1,3,2)
    dim0=torch.cat((ori1,or2),0)
    dim1=torch.cat((ori1,or2),1)
    dim2=torch.cat((ori1,or2),2)
    print(dim2.shape,dim1.shape,dim0.shape)
    dim0 = torch.stack((ori1, or2), 0)
    dim1 = torch.stack((ori1, or2), 1)
    dim2 = torch.stack((ori1, or2), 2)
    print(dim2.shape,dim1.shape,dim0.shape)

    t1 = torch.randn(3,4,5)
    t2 = torch.randn(3,5)
    print(t1,t2)
