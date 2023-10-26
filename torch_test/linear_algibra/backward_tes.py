import torch

if __name__=="__main__":
    a = torch.ones([2, 1],requires_grad=True)
    y = torch.zeros([1,2])
    y[0,1]= 2*(a[1,0]**2+a[0,0]**2)
    y[0,0]= 2*(a[1,0]*a[0,0])
    y.backward(torch.FloatTensor([[1, 0]]),retain_graph=True)
    print(a.grad)
    a.grad.data.zero_()
    y.backward(torch.FloatTensor([[0, 1]]),retain_graph=True)
    print(a.grad)
    a.grad.data.zero_()
    print(torch.autograd.grad(y.sum(),a[0,0],allow_unused=True))