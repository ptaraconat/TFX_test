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

def get_collocation_points(n_points, 
                           space_domainx = 2,
                           space_domainy = 2, 
                           time_domain = 1, 
                           regular_sampling = True):
    '''
    arguments 
    n_points ::: int ::: number of collocation points 
    space_domainx ::: float ::: physical size/length of the domain along x coords
    space_domainy ::: float ::: physical size/length of the domain along y coords
    time_domain ::: float ::: physical time domain length/duration 
    regular_sampling ::: bool ::: whether or not the collocation point 
    are random or regular
    returns 
    x ::: torch.tensor :::
    t ::: torch.tensor ::: 
    '''
    if regular_sampling :
        N = ceil(n_points**(1/3))
        #
        t_tmp = torch.linspace(0, time_domain, N)
        x_tmp = torch.linspace(-0.5*space_domainx, 0.5*space_domainx, N)
        y_tmp = torch.linspace(-0.5*space_domainy, 0.5*space_domainy, N)
        # calculate (x,t) grid
        x_grid, y_grid, t_grid = torch.meshgrid([x_tmp, y_tmp, t_tmp])
        # flatten grid 
        x = x_grid.flatten()
        y = y_grid.flatten()
        t = t_grid.flatten()  
    else : 
        # to continue ....
        N = n_points
    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True     
    return torch.unsqueeze(x, 1), torch.unsqueeze(y, 1), torch.unsqueeze(t, 1)

def get_initial_condition_data(n_points,
                               space_domainx = 2, 
                               space_domainy = 2,
                               regular_sampling = True):
    '''
    arguments 
    n_points ::: int ::: number of collocation points 
    space_domainx ::: float ::: physical size/length of the domain along x coords
    space_domainy ::: float ::: physical size/length of the domain along y coords
    regular_sampling ::: bool :::whether or not the collocation point 
    are random or regular
    returns 
    x ::: torch.tensor ::: x coordinates of the data associated to init conditions 
    y ::: torch.tensor :::
    t ::: torch.tensor ::: time values 
    u ::: torch.tensor ::: Solved variable values at time = 0 
    '''
    if regular_sampling : 
        N = ceil(n_points**(1/2))
        x_tmp = torch.linspace(-0.5*space_domainx, 0.5*space_domainx, N)
        y_tmp = torch.linspace(-0.5*space_domainy, 0.5*space_domainy, N)
        x_grid, y_grid = torch.meshgrid([x_tmp,y_tmp])
        x = x_grid.flatten()
        y = y_grid.flatten()
        t = torch.zeros_like(x)
    u = torch.cos(torch.pi*0.5*x)*torch.cos(torch.pi*0.5*y)
    return x, y, t, u

def get_boundary_condition_data(n_points, 
                                space_domainx = 2,
                                space_domainy = 2,
                                time_domain = 1,
                                regular_sampling = True):
    '''
    arguments 
    n_points ::: int ::: number of collocation points 
    space_domainx ::: float ::: physical size/length of the domain along x coords
    space_domainy ::: float ::: physical size/length of the domain along y coords
    time_domain ::: float ::: physical time domain length/duration 
    regular_sampling ::: bool ::: whether or not the collocation point 
    are random or regular
    returns 
    x ::: torch.tensor ::: 
    y ::: torch.tensor ::: 
    t ::: torch.tensor :::
    u ::: torch.tensor ::: 
    '''
    if regular_sampling : 
        N = int((n_points/4)**(1/2))
        # Generate 1D spaces for each problem dimension
        x_ls = torch.linspace(-0.5*space_domainx,0.5*space_domainx,N)
        y_ls = torch.linspace(-0.5*space_domainy,0.5*space_domainy,N)
        t_ls = torch.linspace(0, time_domain, N)
        # grid for fixed x planes 
        fx_grid_y, fx_grid_t = torch.meshgrid([y_ls, t_ls])
        # grid for fixed y planes
        fy_grid_x, fy_grid_t = torch.meshgrid([x_ls, t_ls])
        # flatten grids 
        fx_y = fx_grid_y.flatten()
        fx_t = fx_grid_t.flatten()
        fy_x = fy_grid_x.flatten()
        fy_t = fy_grid_t.flatten()
        #
        x1 = -0.5*space_domainx*torch.ones_like(fx_y)
        x2 =  0.5*space_domainx*torch.ones_like(fx_y)
        y1 = -0.5*space_domainy*torch.ones_like(fx_y)
        y2 =  0.5*space_domainy*torch.ones_like(fx_y)
        # gather 4 boundary planes in variables x, y and t 
        x = torch.cat((x1  , x2  , fy_x, fy_x),0)
        y = torch.cat((fx_y, fx_y, y1  , y2),0)
        t = torch.cat((fx_t, fx_t, fy_t, fy_t), 0)
        # Set boundary values 
        u = torch.zeros_like(x)
    return x, y, t, u

def get_training_points(n_point_ic, 
                        n_point_bc,
                        space_domainx = 2, 
                        space_domainy = 2,
                        time_domain = 1, 
                        regular_sampling = True):
    '''
    arguments 
    n_points_ic ::: int ::: number of data point associated with 
    initial condition 
    n_points_bc ::: int ::: number of data point associated with 
    boundary conditions 
    space_domain ::: float ::: physical size/length of the domain  
    time_domain ::: float ::: physical time domain length/duration 
    regular_sampling ::: bool ::: whether or not the collocation point 
    are random or regular
    returns 
    x ::: torch.tensor ::: 
    y ::: torch.tensor ::: 
    t ::: torch.tensor :::
    u ::: torch.tensor :::
    '''
    # Get initial & boundary data points and associated values 
    xi, yi, ti, ui = get_initial_condition_data(n_point_ic, 
                                                space_domainx=space_domainx, 
                                                space_domainy=space_domainy,
                                                regular_sampling=regular_sampling)
    xbc, ybc, tbc, ubc = get_boundary_condition_data(n_point_bc, 
                                                     space_domainx=space_domainx,
                                                     space_domainy=space_domainy, 
                                                     time_domain=time_domain,
                                                     regular_sampling=regular_sampling)
    # Gather initial & boundary conditions
    x = torch.cat((xi, xbc), 0)
    y = torch.cat((yi, ybc), 0)
    t = torch.cat((ti, tbc), 0)
    u = torch.cat((ui, ubc), 0)
    return torch.unsqueeze(x, 1), torch.unsqueeze(y, 1), torch.unsqueeze(t, 1), torch.unsqueeze(u, 1)

###
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
    u_yy = torch.autograd.grad(u_y,y, 
                               grad_outputs = torch.ones_like(u_y), 
                               create_graph = True,
                               retain_graph = True)[0]
    alpha = 1.
    heat_eq_residual = alpha**2*(u_xx+u_yy) - u_t
    return heat_eq_residual
###

def get_pde_loss(x,y,t,model):
    '''
    arguments 
    x ::: torch.tensor ::: colocation points x coordinates 
    y ::: torch.tensor ::: colocation points y coordinates 
    t ::: torch.tensor ::: colocation point time coordinates
    model ::: torch.nn ::: Neural network
    returns 
    loss
    '''
    # Neural Network predictions 
    input_data = torch.cat((x,y,t), dim = 1)
    u_pred = model(input_data)
    # Calculate gradients and time differenciation
    u_x = torch.autograd.grad(u_pred,x,
                              grad_outputs = torch.ones_like(u_pred), 
                              create_graph = True,
                              retain_graph = True)[0]
    u_y = torch.autograd.grad(u_pred,y,
                              grad_outputs = torch.ones_like(u_pred), 
                              create_graph = True,
                              retain_graph = True)[0]
    u_t = torch.autograd.grad(u_pred,t,
                              grad_outputs = torch.ones_like(u_pred), 
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
    # Calculate the residual (PDE expression)
    alpha = 1.
    residual = alpha**2*(u_xx+u_yy) - u_t
    # Calculate PDE associated loss 
    loss = nn.MSELoss()(residual,torch.zeros_like(residual))
    
    return loss

def get_data_loss(x,y,t,u,model):
    '''
    arguments 
    x ::: torch.tensor ::: x coordinates of the data 
    x ::: torch.tensor ::: y coordinates of the data
    t ::: torch.tensor ::: time values of the data  
    u ::: torch.tensor ::: Expected values 
    model ::: torch.nn ::: 
    returns 
    loss
    '''
    # NN prediction 
    input_data = torch.cat((x,y,t), dim = 1)
    u_pred = model(input_data)
    # Calculate Loss 
    loss = nn.MSELoss()(u, u_pred)
    return loss

def loss_func(x_col,
              y_col,
              t_col,
              x_data,
              y_data,
              t_data,
              u_data,
              model):
    '''
    arguments :
    x_col ::: torch.tensor ::: x coordinates of collocated point
    y_col ::: torch.tensor ::: y coordinates of collocated point
    t_col ::: torch.tensor ::: t coordinate of collocated points 
    x_data ::: torch.tensor ::: x coordinates of data points 
    y_data ::: torch.tensor ::: y coordinates of data points 
    t_data ::: torch.tensor ::: t coordinates of data points 
    u_data ::: toorch.tensor ::: expected values for data points 
    model ::: torch.nn ::: Nueral Network model 
    returns :
    loss ::: torch.tensor (1) ::: total loss : pde loss + data loss 
    '''
    loss_pde = get_pde_loss(x_col,y_col,t_col,model)
    loss_data = get_data_loss(x_data,y_data,t_data,u_data, model)
    loss = loss_pde+loss_data
    return loss 

class PINN_Trainer:
    '''
    '''
    def __init__(self, space_domainx = 2, space_domainy = 2, time_domain = 1, n_col_points = 300, n_ic_points = 50, n_bc_points = 100):
        '''
        '''
        self.space_domainx = space_domainx
        self.space_domainy = space_domainy
        self.time_domain = time_domain
        self.n_col_points = n_col_points
        self.n_ic_points = n_ic_points
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
        self.x_col, self.y_col, self.t_col = get_collocation_points(n_col_points, 
                                                                    space_domainx = space_domainx,
                                                                    space_domainy = space_domainy, 
                                                                    time_domain = time_domain)
        self.x_data, self.y_data, self.t_data, self.u_data = get_training_points(n_bc_points,
                                                                                 n_ic_points, 
                                                                                 space_domainx = space_domainx, 
                                                                                 space_domainy = space_domainy,
                                                                                 time_domain = time_domain)
        print(self.x_col.shape,self.y_col.shape,self.t_col.shape)
        print(self.x_data.shape,self.y_data.shape,self.t_data.shape,self.u_data.shape)
        #
        self.iter = 1
    
    def calc_loss(self):
        '''
        '''
        self.adam.zero_grad()
        self.lbfgs.zero_grad()
        loss = loss_func(self.x_col,
                         self.y_col,
                         self.t_col,
                         self.x_data,
                         self.y_data,
                         self.t_data,
                         self.u_data,
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
    print('hey')
    trainer = PINN_Trainer(space_domainx = 2, 
                           space_domainy = 2, 
                           time_domain = 1, 
                           n_col_points = 1000, 
                           n_ic_points = 400, 
                           n_bc_points = 600)
    trainer.train()
    #
    model = trainer.neural_network
    x_vals = torch.linspace(-1,1,100)
    y_vals = torch.linspace(-1,1,100)
    X,Y = torch.meshgrid(x_vals,y_vals)
    with torch.no_grad() : 
        t_vals = torch.ones_like(X) * 0 
        input_data = torch.stack([X.flatten(),Y.flatten(),t_vals.flatten()], dim = 1)
        solution = model(input_data).reshape(X.shape, Y.shape)
        plt.figure(figsize = (8,6))
        sns.heatmap(solution, cmap = 'jet')
        plt.savefig('t0_2Dheat.png', format = 'png')
        plt.close()

    with torch.no_grad() : 
        t_vals = torch.ones_like(X) * 0.1 
        input_data = torch.stack([X.flatten(),Y.flatten(),t_vals.flatten()], dim = 1)
        solution = model(input_data).reshape(X.shape, Y.shape)
        plt.figure(figsize = (8,6))
        sns.heatmap(solution, cmap = 'jet')
        plt.savefig('t0p1_2Dheat.png', format = 'png')
        plt.close()

    with torch.no_grad() : 
        t_vals = torch.ones_like(X) * 0.2 
        input_data = torch.stack([X.flatten(),Y.flatten(),t_vals.flatten()], dim = 1)
        solution = model(input_data).reshape(X.shape, Y.shape)
        plt.figure(figsize = (8,6))
        sns.heatmap(solution, cmap = 'jet')
        plt.savefig('t0p2_2Dheat.png', format = 'png')
        plt.close()

    with torch.no_grad() : 
        t_vals = torch.ones_like(X) * 0.9 
        input_data = torch.stack([X.flatten(),Y.flatten(),t_vals.flatten()], dim = 1)
        solution = model(input_data).reshape(X.shape, Y.shape)
        plt.figure(figsize = (8,6))
        sns.heatmap(solution, cmap = 'jet')
        plt.savefig('t0p9_2Dheat.png', format = 'png')
        plt.close()