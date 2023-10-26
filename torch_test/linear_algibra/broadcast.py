import torch
if __name__=="__main__":
    t1 = torch.arange(0,60,1).reshape(3,4,5)
    t2 = torch.arange(0,15,1).reshape(3,5)
    print(t1, t2)
    t2_un_squeeze = torch.unsqueeze(t2, 1)
    print(t2_un_squeeze.shape)
    print(t1+t2_un_squeeze)

