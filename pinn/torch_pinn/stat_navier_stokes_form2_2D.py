import torch 
import torch.nn as nn 
import torch.optim as optim
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from math import * 

class PINN(nn.Module):
    '''
    '''
    def __init__(self):
        '''
        '''
        super(PINN, self).__init__()
        self.nn = nn.Sequential(nn.Linear(2,64),
                                nn.Tanh(),
                                nn.Linear(64,64),
                                nn.Tanh(),
                                nn.Linear(64,2)
                                )
                
    def forward(self,X):
        '''
        arguments 
        X ::: torch.tensor (n_batch, 2) ::: indput data 
        '''
        return self.nn(X)

def get_pde_loss(x,y,model):
    '''
    arguments 
    x ::: torch.tensor ::: colocation points x coordinates 
    y ::: torch.tensor ::: colocation points y coordinates 
    model ::: torch.nn ::: Neural network
    returns 
    loss
    '''
    # Neural Network predictions 
    input_data = torch.cat((x,y), dim = 1)
    res = model(input_data)
    #
    phi = res[:,0].unsqueeze(dim = 1)
    p = res[:,1].unsqueeze(dim = 1)
    #
    u = torch.autograd.grad(phi,y,
                            grad_outputs = torch.ones_like(phi), 
                            create_graph = True,
                            retain_graph = True)[0]
    v = -torch.autograd.grad(phi,x,
                             grad_outputs = torch.ones_like(phi), 
                             create_graph = True,
                             retain_graph = True)[0]
    # Calculate gradients and time differenciation
    u_x = torch.autograd.grad(u,x,
                              grad_outputs = torch.ones_like(u), 
                              create_graph = True,
                              retain_graph = True)[0]
    u_y = torch.autograd.grad(u,y,
                              grad_outputs = torch.ones_like(u), 
                              create_graph = True,
                              retain_graph = True)[0]
    v_x = torch.autograd.grad(v,x,
                              grad_outputs = torch.ones_like(v), 
                              create_graph = True,
                              retain_graph = True)[0]
    v_y = torch.autograd.grad(v,y,
                              grad_outputs = torch.ones_like(v), 
                              create_graph = True,
                              retain_graph = True)[0]
    p_x = torch.autograd.grad(p,x,
                              grad_outputs = torch.ones_like(p), 
                              create_graph = True,
                              retain_graph = True)[0]
    p_y = torch.autograd.grad(p,y,
                              grad_outputs = torch.ones_like(p), 
                              create_graph = True,
                              retain_graph = True)[0]
    u_xx = torch.autograd.grad(u_x,x,
                              grad_outputs = torch.ones_like(u_x), 
                              create_graph = True,
                              retain_graph = True)[0]
    u_yy = torch.autograd.grad(u_y,y,
                              grad_outputs = torch.ones_like(u_y), 
                              create_graph = True,
                              retain_graph = True)[0]
    v_xx = torch.autograd.grad(v_x,x,
                              grad_outputs = torch.ones_like(v_x), 
                              create_graph = True,
                              retain_graph = True)[0]
    v_yy = torch.autograd.grad(v_y,y,
                              grad_outputs = torch.ones_like(v_y), 
                              create_graph = True,
                              retain_graph = True)[0]
    # Calculate the residual (PDE expression)
    rho = 1
    nu = 1e-2
    residual_x = u*u_x + v*u_y + p_x/rho - (nu/rho)*(u_xx + u_yy)
    residual_y = u*v_x + v*v_y + p_y/rho - (nu/rho)*(v_xx + v_yy)
    # Calculate PDE associated loss 
    loss_x = nn.MSELoss()(residual_x,torch.zeros_like(residual_x))
    loss_y = nn.MSELoss()(residual_y,torch.zeros_like(residual_y))
    
    return loss_x + loss_y  

def get_velocity_data_loss(x,y,u,v,model):
    '''
    arguments 
    x ::: torch.tensor ::: x coordinates of the data 
    y ::: torch.tensor ::: y coordinates of the data
    u ::: torch.tensor ::: Expected x velocity component values 
    v ::: torch.tensor ::: Expected y velocity component values 
    model ::: torch.nn ::: 
    returns 
    loss
    '''
    # Neural Network predictions 
    input_data = torch.cat((x,y), dim = 1)
    res = model(input_data)
    #
    phi = res[:,0].unsqueeze(dim = 1)
    p = res[:,1].unsqueeze(dim = 1)
    #
    u_pred = torch.autograd.grad(phi,y,
                                 grad_outputs = torch.ones_like(phi), 
                                 create_graph = True,
                                 retain_graph = True)[0]
    v_pred = -torch.autograd.grad(phi,x,
                                  grad_outputs = torch.ones_like(phi), 
                                  create_graph = True,
                                  retain_graph = True)[0]
    # Calculate Loss 
    predictions = torch.cat((u_pred,v_pred),dim = 1)
    expected = torch.cat((u,v), dim = 1)
    loss = nn.MSELoss()(expected, predictions)
    return loss

def get_pressure_data_loss(x,y,p,model):
    '''
    arguments 
    x ::: torch.tensor ::: x coordinates of the data 
    y ::: torch.tensor ::: y coordinates of the data
    p ::: torch.tensor ::: Expected pressure values
    model ::: torch.nn ::: 
    returns 
    loss
    '''
    # Neural Network predictions 
    input_data = torch.cat((x,y), dim = 1)
    res = model(input_data)
    #
    phi = res[:,0].unsqueeze(dim = 1)
    p_pred = res[:,1].unsqueeze(dim = 1)
    # Calculate Loss 
    loss = nn.MSELoss()(p, p_pred)
    return loss

def loss_func(x_col,
              y_col,
              x_vel_data,
              y_vel_data,
              u_vel_data,
              v_vel_data,
              x_pres_data,
              y_pres_data,
              p_pres_data,
              model):
    '''
    arguments :
    x_col ::: torch.tensor ::: x coordinates of collocated point
    y_col ::: torch.tensor ::: y coordinates of collocated point
    x_data ::: torch.tensor ::: x coordinates of data points 
    y_data ::: torch.tensor ::: y coordinates of data points 
    u_data ::: toorch.tensor ::: 
    v_data ::: toorch.tensor ::: 
    p_data ::: toorch.tensor ::: 
    model ::: torch.nn ::: Nueral Network model 
    returns :
    loss ::: torch.tensor (1) ::: total loss : pde loss + data loss 
    '''
    loss_pde = get_pde_loss(x_col, y_col, model)
    loss_vel_data = get_velocity_data_loss(x_vel_data, y_vel_data, u_vel_data, v_vel_data, model)
    #loss_pres_data = get_pressure_data_loss(x_pres_data, y_pres_data, p_pres_data, model)
    loss = loss_pde+loss_vel_data
    return loss 

def get_collocation_points(n_points, 
                           space_domainx = 2,
                           space_domainy = 2,  
                           regular_sampling = True):
    '''
    arguments 
    n_points ::: int ::: number of collocation points 
    space_domainx ::: float ::: physical size/length of the domain along x coords
    space_domainy ::: float ::: physical size/length of the domain along y coords
    regular_sampling ::: bool ::: whether or not the collocation point 
    are random or regular
    returns 
    x ::: torch.tensor :::
    y ::: torch.tensor ::: 
    '''
    if regular_sampling :
        N = ceil(n_points**(1/2))
        #
        x_tmp = torch.linspace(-0.5*0.98*space_domainx, 0.5*0.98*space_domainx, N)
        y_tmp = torch.linspace(-0.5*0.98*space_domainy, 0.5*0.98*space_domainy, N)
        # calculate (x,y) grid
        x_grid, y_grid = torch.meshgrid([x_tmp, y_tmp])
        # flatten grid 
        x = x_grid.flatten()
        y = y_grid.flatten() 
    else : 
        # to continue ....
        N = n_points
    x.requires_grad = True
    y.requires_grad = True   
    return torch.unsqueeze(x, 1), torch.unsqueeze(y, 1)

def get_velocity_condition_data(n_points, 
                                space_domainx = 2,
                                space_domainy = 2,
                                regular_sampling = True):
    '''
    arguments 
    n_points ::: int ::: number of collocation points 
    space_domainx ::: float ::: physical size/length of the domain along x coords
    space_domainy ::: float ::: physical size/length of the domain along y coords
    regular_sampling ::: bool ::: whether or not the collocation point 
    are random or regular
    returns 
    x ::: torch.tensor ::: 
    y ::: torch.tensor ::: 
    u ::: torch.tensor ::: 
    v ::: torch.tensor :::  
    '''
    if regular_sampling : 
        N = int((n_points/3))
        # Generate 1D spaces for each problem dimension
        x_ls = torch.linspace(-0.5*space_domainx,0.5*space_domainx,N)
        y_ls = torch.linspace(-0.5*(space_domainy*0.99),0.5*(space_domainy*0.99),N)
        #
        x1 = -0.5*space_domainx*torch.ones_like(x_ls)
        #x2 =  0.5*space_domainx*torch.ones_like(x_ls)
        y1 = -0.5*space_domainy*torch.ones_like(x_ls)
        y2 =  0.5*space_domainy*torch.ones_like(x_ls)
        #
        velocity = 0.05
        u1 = velocity*torch.ones_like(x1)
        u_ew = torch.zeros_like(x1)
        v_ew = torch.zeros_like(x1)
        # gather 4 boundary planes in variables x, y and t 
        x = torch.cat((x1, x_ls, x_ls),0)
        y = torch.cat((y_ls, y1, y2),0)
        # Set boundary values 
        u = torch.cat((u1, u_ew, u_ew), 0)
        v = torch.cat((v_ew, v_ew, v_ew), 0)
    x.requires_grad = True
    y.requires_grad = True  
    return torch.unsqueeze(x, 1), torch.unsqueeze(y, 1), torch.unsqueeze(u, 1), torch.unsqueeze(v, 1)

def get_pressure_condition_data(n_points, 
                                space_domainx = 2,
                                space_domainy = 2,
                                regular_sampling = True):
    '''
    arguments 
    n_points ::: int ::: number of collocation points 
    space_domainx ::: float ::: physical size/length of the domain along x coords
    space_domainy ::: float ::: physical size/length of the domain along y coords
    regular_sampling ::: bool ::: whether or not the collocation point 
    are random or regular
    returns 
    x ::: torch.tensor ::: 
    y ::: torch.tensor ::: 
    p ::: torch.tensor ::: 
    '''
    if regular_sampling : 
        N = int(n_points)
        # Generate 1D spaces for each problem dimension
        y_ls = torch.linspace(-0.5*(space_domainy),0.5*(space_domainy),N)
        #
        x2 =  0.5*space_domainx*torch.ones_like(y_ls)
        # Set boundary pressure value 
        pressure = 0 
        p2 = pressure*torch.ones_like(x2)
    x2.requires_grad = True
    y_ls.requires_grad = True 
    return torch.unsqueeze(x2, 1), torch.unsqueeze(y_ls, 1), torch.unsqueeze(p2, 1)

class PINN_Trainer:
    '''
    '''
    def __init__(self, space_domainx = 2, space_domainy = 2, n_col_points = 300, n_bc_points = 100):
        '''
        '''
        self.space_domainx = space_domainx
        self.space_domainy = space_domainy
        self.n_col_points = n_col_points
        self.n_bc_points = n_bc_points
        #
        self.neural_network = PINN()
        #
        self.adam = torch.optim.Adam(self.neural_network.parameters())
        self.lbfgs = torch.optim.LBFGS(self.neural_network.parameters(),
                                       lr = 1., 
                                       max_iter = 50000, 
                                       max_eval = 50000, 
                                       history_size = 50, 
                                       tolerance_grad = 1e-7, 
                                       tolerance_change = 1.0*np.finfo(float).eps, 
                                       line_search_fn = 'strong_wolfe')
        #
        self.x_col, self.y_col = get_collocation_points(n_col_points, 
                                                        space_domainx = space_domainx,
                                                        space_domainy = space_domainy)
        self.x_vel_data, self.y_vel_data, self.u_vel_data, self.v_vel_data = get_velocity_condition_data(self.n_bc_points*0.75, 
                                                                                                         space_domainx = space_domainx,
                                                                                                         space_domainy = space_domainy)
        self.x_pres_data, self.y_pres_data, self.p_pres_data = get_pressure_condition_data(self.n_bc_points*0.25,
                                                                                           space_domainx = space_domainx,
                                                                                           space_domainy = space_domainy)
        x_col = self.x_col.detach().numpy()
        y_col = self.y_col.detach().numpy()
        x_vel_data = self.x_vel_data.detach().numpy()
        y_vel_data = self.y_vel_data.detach().numpy()
        x_pres_data = self.x_pres_data.detach().numpy()
        y_pres_data = self.y_pres_data.detach().numpy()
        plt.scatter(x_col,y_col)
        plt.scatter(x_vel_data, y_vel_data, c = 'r')
        plt.scatter(x_pres_data, y_pres_data, c = 'k')
        plt.grid()
        plt.savefig('test.png', format = 'png')
        plt.close()
        #
        self.iter = 1
    
    def calc_loss(self):
        '''
        '''
        self.adam.zero_grad()
        self.lbfgs.zero_grad()
        loss = loss_func(self.x_col,
                         self.y_col,
                         self.x_vel_data,
                         self.y_vel_data,
                         self.u_vel_data,
                         self.v_vel_data,
                         self.x_pres_data,
                         self.y_pres_data,
                         self.p_pres_data,
                         self.neural_network)
        loss.backward()
        # Display loss in console
        if self.iter % 100 == 0 :
            print('ITER ', self.iter, '; LOSS : ', loss)
        self.iter +=1
        return loss
    
    def train(self):
        '''
        '''
        #Perform some ADAM step 
        for _ in range(1000):
            self.adam.step(self.calc_loss)
        #Continue with LBFGS 
        self.lbfgs.step(self.calc_loss)

if __name__ == '__main__':
    space_domainx = 5
    space_domainy = 1
    n_col_points = 1000
    n_bc_points = 100

    trainer = PINN_Trainer(space_domainx = space_domainx, 
                           space_domainy = space_domainy, 
                           n_col_points = n_col_points, 
                           n_bc_points = n_bc_points)
    trainer.train()
    #
    model = trainer.neural_network
    x_vals = torch.linspace(-space_domainx*0.5,space_domainx*0.5,100)
    y_vals = torch.linspace(-space_domainy*0.5,space_domainy*0.5,100)
    X,Y = torch.meshgrid(x_vals,y_vals)
    x = X.flatten()
    y = Y.flatten()
    x.requires_grad = True
    y.requires_grad = True 
    input_data = torch.stack([x,y], dim = 1)
    #
    res = model(input_data)
    phi = res[:,0].unsqueeze(dim = 1)
    p = res[:,1].unsqueeze(dim = 1)
    #
    u = torch.autograd.grad(phi,y,
                            grad_outputs = torch.ones_like(phi), 
                            create_graph = True,
                            retain_graph = True)[0]
    v = -torch.autograd.grad(phi,x,
                             grad_outputs = torch.ones_like(phi), 
                             create_graph = True,
                             retain_graph = True)[0]
    u = u.reshape(X.shape, Y.shape)
    v = v.reshape(X.shape, Y.shape)
    p = p.reshape(X.shape, Y.shape)
    #
    plt.figure(figsize = (8,6))
    sns.heatmap(u.detach().numpy(), cmap = 'jet')
    plt.savefig('stead_NS_2D_ux.png', format = 'png')
    plt.close()
    #
    plt.figure(figsize = (8,6))
    sns.heatmap(v.detach().numpy(), cmap = 'jet')
    plt.savefig('stead_NS_2D_uy.png', format = 'png')
    plt.close()
    #
    plt.figure(figsize = (8,6))
    sns.heatmap(p.detach().numpy(), cmap = 'jet')
    plt.savefig('stead_NS_2D_p.png', format = 'png')
    plt.close()
    