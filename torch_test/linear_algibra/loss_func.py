
import torch
import torch.nn as nn

if __name__=="__main__":
    mse = nn.MSELoss()
    t1 = torch.randn(5, requires_grad=True)
    t2 = torch.randn(5,requires_grad=True)
    print(mse(t1,t2))

    bce = nn.BCELoss()
    t1s = torch.sigmoid(t1)
    t2s = torch.randint(0,2,(5,)).float()
    print(bce(t1s,t2s))

    bce_with_log = nn.BCEWithLogitsLoss()
    print(bce_with_log(t1,t2s))

    # 多分类预测
    N = 10
    t1n = torch.randn(5,N,requires_grad=True)
    t2n = torch.randint(0,N,(5,))
    print(t2n.shape)
    t1ns = nn.functional.log_softmax(t1n,-1)
    nll = nn.NLLLoss()
    print(nll(t1ns,t2n))
    ce = nn.CrossEntropyLoss()
    print(ce(t1n,t2n))