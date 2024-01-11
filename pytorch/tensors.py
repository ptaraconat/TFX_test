import torch 
import numpy as np 

x = torch.empty(2,3)
print(x)

x = torch.tensor([2.5,2,3])
print(x)
print(x.dtype)
print(x.size())

x = torch.rand(2,2)
y = torch.rand(2,2)
z = x + y 
print(x)
print(y)
print(z)
print(y.add_(x))

x = torch.rand(5,3)
print(x)
print(x[:,0])
print(x[1,:])
# get a specific value from a one element tensor 
val = x[1,1].item()

x = torch.rand(4,4)
y = x.view(16)
print(x)
print(y)
y = x.view(-1,8)
print(y)

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print(type(a),type(b))