import torch 
import torch.nn as nn 
import torch.optim as optim 
import seaborn as sns 
import matplotlib.pyplot as plt 

class PINN(nn.Module):
    '''
    '''
    def __init__(self):
        '''
        '''
        super(PINN, self).__init__()
        self.nn = nn.Sequential(nn.Linear(3,64),
                                nn.Tanh(),
                                nn.Linear(64,64),
                                nn.Tanh(),
                                nn.Linear(64,1)
                                )
        
    def forward(self,X):
        '''
        arguments 
        X ::: torch.tensor (n_batch, 3) ::: indput data 
        '''
        return self.nn(X)

def initial_condition(x,y):
    '''
    arguments 
    x ::: torch.tensor (n_batch,1) ::: x coordinate
    y ::: torch.tensor (n_batch,1) ::: y coordinate
    '''
    return torch.sin(torch.pi*x)*torch.sin(torch.pi*y)

def boundary_condition(x,y,t, cunstom_value):
    '''
    arguments 
    x ::: torch.tensor (n_batch,1) ::: x coordinate
    y ::: torch.tensor (n_batch,1) ::: y coordinate
    t ::: toch.tensor
    '''
    return torch.full_like(x,cunstom_value)

def generate_training_data(num_points):
    x = torch.rand(num_points,1,requires_grad = True)
    y = torch.rand(num_points,1,requires_grad = True)
    t = torch.rand(num_points,1,requires_grad = True)
    return x,y,t

def generate_boundary_points(num_points): 
    x_bnd = torch.tensor([0.0,1.0]).repeat(num_points//2)
    y_bnd = torch.rand(num_points)

    if torch.rand(1) > 0.5 : 
        x_bnd, y_bnd  = y_bnd, x_bnd
    
    return x_bnd.view(-1,1), y_bnd.view(-1,1)

def generate_boundary_training_data(num_points):
    '''
    '''
    x_bnd, y_bnd = generate_boundary_points(num_points)
    t = torch.rand(num_points,1,requires_grad = True)
    return x_bnd, y_bnd, t

def pde(x,y,t,model):
    '''
    '''
    input_data = torch.cat([x,y,t], dim = 1)
    u = model(input_data)
    u_t = torch.autograd.grad(u,t, 
                              grad_outputs = torch.ones_like(u), 
                              create_graph = True,
                              retain_graph = True)[0]
    u_x = torch.autograd.grad(u,x, 
                              grad_outputs = torch.ones_like(u), 
                              create_graph = True,
                              retain_graph = True)[0]
    u_y = torch.autograd.grad(u,y,
                              grad_outputs = torch.ones_like(u), 
                              create_graph = True,
                              retain_graph = True)[0]
    u_xx = torch.autograd.grad(u_x,x, 
                               grad_outputs = torch.ones_like(u_x), 
                               create_graph = True,
                               retain_graph = True)[0]
    u_yy = torch.autograd.grad(u_y,x, 
                               grad_outputs = torch.ones_like(u_y), 
                               create_graph = True,
                               retain_graph = True)[0]
    alpha = 1.
    heat_eq_residual = alpha**2*(u_xx+u_yy) - u_t
    return heat_eq_residual



def train_pinn(model, num_iterations, num_points, 
               optimizer = None):
    '''
    '''
    if optimizer == None : 
        optimizer = optim.Adam(model.parameters(),lr = 1e-3)
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        x,y,t = generate_training_data(num_points)
        x_b, y_b, t_b = generate_boundary_training_data(num_points)

        t_initial = torch.zeros_like(x)
        u_initial = initial_condition(x,y)

        custom_value = 0 
        u_bnd_x = boundary_condition(x_b,y_b,t_b,custom_value)
        u_bnd_y = boundary_condition(y_b,x_b,t_b,custom_value)

        residual = pde(x,y,t,model)

        loss = nn.MSELoss()(u_initial, model(torch.cat([x,y,t_initial], dim = 1))) + \
               nn.MSELoss()(u_bnd_x, model(torch.cat([x_b,y_b,t_b], dim = 1))) + \
               nn.MSELoss()(u_bnd_y, model(torch.cat([x_b,y_b,t_b], dim = 1))) + \
               nn.MSELoss()(residual, torch.zeros_like(residual))
        
        loss.backward()
        optimizer.step()

        if iteration%100 == 0 :
            print("iteration ", iteration, '/ loss :', loss)


if __name__ == '__main__': 
    print('hey')
    num_points = 10
    x,y,t = generate_training_data(num_points)
    print(x,y,t)
    model = PINN()
    res = pde(x,y,t,model)
    print(res)

    model = PINN()
    num_iterations = 10000
    num_points = 1000 
    train_pinn(model,num_iterations,num_points)

    with torch.no_grad() : 
        x_vals = torch.linspace(0,1,100)
        y_vals = torch.linspace(0,1,100)
        X,Y = torch.meshgrid(x_vals,y_vals)
        t_vals = torch.ones_like(X) * 0 
        print(t_vals.shape)
        print(X.flatten().shape,Y.flatten().shape,t_vals.flatten().shape)
        input_data = torch.stack([X.flatten(),Y.flatten(),t_vals.flatten()], dim = 1)
        solution = model(input_data).reshape(X.shape, Y.shape)

        plt.figure(figsize = (8,6))
        sns.heatmap(solution, cmap = 'jet')
        plt.savefig('t0_2Dheat.png', format = 'png')
        plt.close()
    
    with torch.no_grad() : 
        x_vals = torch.linspace(0,1,100)
        y_vals = torch.linspace(0,1,100)
        X,Y = torch.meshgrid(x_vals,y_vals)
        t_vals = torch.ones_like(X) * 0.1
        print(t_vals.shape)
        print(X.flatten().shape,Y.flatten().shape,t_vals.flatten().shape)
        input_data = torch.stack([X.flatten(),Y.flatten(),t_vals.flatten()], dim = 1)
        solution = model(input_data).reshape(X.shape, Y.shape)

        plt.figure(figsize = (8,6))
        sns.heatmap(solution, cmap = 'jet')
        plt.savefig('t0p1_2Dheat.png', format = 'png')
        plt.close()