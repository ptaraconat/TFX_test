import torch 
import torch.nn as nn 

class LinearRegression(nn.Module):
	def __init__(self,input_dim, output_dim):
		super(LinearRegression, self).__init__()
		# define layers 
		self.lin = nn.Linear(input_dim, output_dim)
	def forward(self,x):
		return self.lin(x)

# Data 
X = torch.tensor([[1],[2],[3],[4]], dtype = torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype = torch.float32)
X_test = torch.tensor([5],dtype = torch.float32)
n_samples, n_features = X.shape
print(n_samples, n_features)
# define forward pass 
input_size = n_features 
output_size = n_features 
#model = nn.Linear(input_size, output_size)
model = LinearRegression(input_size, output_size)
print(f'Prediction before training: f(5) = {model(X_test)}:.3f')
# Loss function 
loss = nn.MSELoss()
# Define optimizer 
learning_rate = 0.01 
n_iters = 200 
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
# Training loop 
for epoch in range(n_iters):
	#
	y_pred = model(X)
	l = loss(Y,y_pred)
	l.backward()
	optimizer.step()
	optimizer.zero_grad()
	if epoch % 10 == 0 : 
		[w,b] = model.parameters()
		print(f'epoch {epoch + 1}: w = {w[0][0]:.3f}, loss = {l:.8f}')