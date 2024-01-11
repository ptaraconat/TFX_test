import torch 

x = torch.randn(4,requires_grad = True)
print(x)
y = x*x+2 
print(y)
y = y.sum()
print(y)
y.backward()
print(x.grad)