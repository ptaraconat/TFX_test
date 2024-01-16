import torch 
import torch.nn as nn 
import numpy as np 

def cross_entropy(actual, predicted):
	loss = -np.sum(actual * np.log(predicted))
	return loss

Y = np.array([1,0,0])

Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.5])

l1 = cross_entropy(Y,Y_pred_good)
l2 = cross_entropy(Y,Y_pred_bad)
print(l1)
print(l2)

### pytorch implementation 
loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])
# n samples x n_classes = 1x3
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]]) # these are logit, must not apply softmax
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred_good,Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

# get class predicted 
_, predictions1 = torch.max(Y_pred_good,1)
_, predictions2 = torch.max(Y_pred_bad,1)

print(predictions1)
print(predictions2)