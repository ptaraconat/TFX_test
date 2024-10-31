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
        self.nn = nn.Sequential(nn.Linear(2,20),
                                nn.Tanh(),
                                nn.Linear(20,30),
                                nn.Tanh(),
                                nn.Linear(30,30),
                                nn.Tanh(),
                                nn.Linear(30,20),
                                nn.Tanh(),
                                nn.Linear(20,20),
                                nn.Tanh(),
                                nn.Linear(20,1)
                                )
        
    def forward(self,X):
        '''
        arguments 
        X ::: torch.tensor (n_batch, 3) ::: indput data 
        '''
        return self.nn(X)

def get_collocation_points(n_points, 
                           space_domain = 2, 
                           time_domain = 1, 
                           regular_sampling = True):
    '''
    arguments 
    n_points ::: int ::: number of collocation points 
    space_domain ::: float ::: physical size/length of the domain 
    time_domain ::: float ::: physical time domain length/duration 
    regular_sampling ::: bool ::: whether or not the collocation point 
    are random or regular
    returns 
    x ::: torch.tensor :::
    t ::: torch.tensor ::: 
    '''
    if regular_sampling :
        N = ceil(np.sqrt(n_points))
        #
        t_tmp = torch.linspace(0, time_domain, N)
        x_tmp = torch.linspace(-0.5*space_domain, 0.5*space_domain, N)
        # calculate (x,t) grid
        x_grid, t_grid = torch.meshgrid([x_tmp, t_tmp])
        # flatten grid 
        # we first use transpose so that the data appears 
        # in a chronological order (from t = 0 to t = time_domain)
        x = torch.transpose(x_grid,0,1).flatten()
        t = torch.transpose(t_grid,0,1).flatten()  
    else : 
        # to continue ....
        N = n_points
        x = torch.rand(N,1,requires_grad = True)
        t = torch.rand(N,1,requires_grad = True) 
    x.requires_grad = True
    t.requires_grad = True     
    return torch.unsqueeze(x, 1), torch.unsqueeze(t, 1)

def get_initial_condition_data(n_points,
                               space_domain = 2, 
                               regular_sampling = True):
    '''
    arguments 
    n_points ::: int ::: number of collocation points 
    space_domain ::: float ::: physical size/length of the domain 
    regular_sampling ::: bool :::whether or not the collocation point 
    are random or regular
    returns 
    x ::: torch.tensor ::: x coordinates of the data associated to init conditions 
    t ::: torch.tensor ::: time values 
    u ::: torch.tensor ::: Solved variable values at time = 0 
    '''
    if regular_sampling : 
        N = n_points
        x = torch.linspace(-0.5*space_domain, 0.5*space_domain, N)
        t = torch.zeros(N)
    u = -torch.sin(torch.pi*x)
    return x, t, u

def get_boundary_condition_data(n_points, 
                                space_domain = 2,
                                time_domain = 1,
                                regular_sampling = True):
    '''
    arguments 
    n_points ::: int ::: number of collocation points 
    space_domain ::: float ::: physical size/length of the domain  
    time_domain ::: float ::: physical time domain length/duration 
    regular_sampling ::: bool ::: whether or not the collocation point 
    are random or regular
    returns 
    x ::: torch.tensor ::: 
    y ::: torch.tensor ::: 
    u ::: torch.tensor ::: 
    '''
    N = int(n_points/2)
    x1 = -0.5*space_domain*torch.ones(N)
    x2 = 0.5*space_domain*torch.ones(N)
    t_tmp = torch.linspace(0, time_domain, N)
    t = torch.cat((t_tmp,t_tmp), 0)
    x = torch.cat((x1,x2),0)
    u = torch.zeros(2*N)
    return x, t, u

def get_training_points(n_point_ic, 
                        n_point_bc,
                        space_domain = 2, 
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
    u ::: torch.tensor :::
    '''
    xi, ti, ui = get_initial_condition_data(n_point_ic, 
                                            space_domain=space_domain, 
                                            regular_sampling=regular_sampling)
    xbc, tbc, ubc = get_boundary_condition_data(n_point_bc, 
                                            space_domain=space_domain, 
                                            time_domain=time_domain,
                                            regular_sampling=regular_sampling)
    #
    x = torch.cat((xi,xbc), 0)
    t = torch.cat((ti,tbc), 0)
    u = torch.cat((ui,ubc), 0)
    return torch.unsqueeze(x, 1), torch.unsqueeze(t, 1), torch.unsqueeze(u, 1)

def get_pde_loss(x,t,model):
    '''
    arguments 
    x ::: torch.tensor ::: colocation points x coordinates 
    t ::: torch.tensor ::: colocation point time coordinates
    model ::: torch.nn ::: Neural network
    returns 
    loss
    '''
    # Neural Network predictions 
    input_data = torch.cat((x,t), dim = 1)
    u_pred = model(input_data)
    # Calculate gradients and time differenciation
    u_x = torch.autograd.grad(u_pred,x,
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
    # Calculate the residual (PDE expression)
    nu = 0.01/torch.pi
    residual = u_t+u_pred*u_x - nu*u_xx
    # Calculate PDE associated loss 
    loss = nn.MSELoss()(residual,torch.zeros_like(residual))
    
    return loss

def get_data_loss(x,t,u,model):
    '''
    arguments 
    x ::: torch.tensor ::: x coordinates of the data 
    t ::: torch.tensor ::: time values of the data  
    u ::: torch.tensor ::: Expected values 
    model ::: torch.nn ::: 
    returns 
    loss
    '''
    # NN prediction 
    input_data = torch.cat((x,t), dim = 1)
    u_pred = model(input_data)
    # Calculate Loss 
    loss = nn.MSELoss()(u, u_pred)
    return loss

def loss_func(x_col,t_col,x_data,t_data,u_data,model):
    '''
    arguments :
    x_col ::: torch.tensor ::: x coordinates of collocated point
    t_col ::: torch.tensor ::: t coordinate of collocated points 
    x_data ::: torch.tensor ::: x coordinates of data points 
    t_data ::: torch.tensor ::: t coordinates of data points 
    u_data ::: toorch.tensor ::: expected values for data points 
    model ::: torch.nn ::: Nueral Network model 
    returns :
    loss ::: torch.tensor (1) ::: total loss : pde loss + data loss 
    '''
    loss_pde = get_pde_loss(x_col,t_col,model)
    loss_data = get_data_loss(x_data,t_data,u_data, model)
    loss = loss_pde+loss_data
    return loss 

class PINN_Trainer:
    '''
    '''
    def __init__(self, space_domain = 2, time_domain = 1, n_col_points = 300, n_ic_points = 50, n_bc_points = 100):
        '''
        '''
        self.space_domain = space_domain
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
        self.x_col, self.t_col = get_collocation_points(n_col_points, 
                                                        space_domain = space_domain, 
                                                        time_domain = time_domain)
        self.x_data, self.t_data, self.u_data = get_training_points(n_bc_points,
                                                                    n_ic_points, 
                                                                    space_domain = space_domain, 
                                                                    time_domain = time_domain)
        print(self.x_col.shape,self.t_col.shape)
        print(self.x_data.shape,self.t_data.shape,self.u_data.shape)
        #
        self.iter = 1
    
    def calc_loss(self):
        '''
        '''
        self.adam.zero_grad()
        self.lbfgs.zero_grad()
        loss = loss_func(self.x_col,self.t_col,self.x_data,self.t_data,self.u_data,self.neural_network)
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
        for i in range(1000):
            self.adam.step(self.calc_loss)
        #Continue with LBFGS 
        self.lbfgs.step(self.calc_loss)

if __name__ == '__main__': 
    trainer = PINN_Trainer(space_domain = 2, 
                           time_domain = 1, 
                           n_col_points = 300, 
                           n_ic_points = 50, 
                           n_bc_points = 100)
    trainer.train()
    h = 0.01
    k = 0.01
    x = torch.arange(-1,1,h)
    t = torch.arange(0,1,k)
    X = torch.stack(torch.meshgrid(x,t)).reshape(2,-1).T
    model =trainer.neural_network
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        y_pred = y_pred.reshape(len(x),len(t)).cpu().numpy()
    print(y_pred)

    sns.set_style('white')
    plt.figure(figsize = (5,3), dpi = 3000 )
    sns.heatmap(y_pred, cmap = 'jet')
    plt.show()
    plt.savefig('burgers1D_colormap.png',format = 'png')
    plt.close()

    plt.plot(y_pred[:,0], 'k-', label = 'init')
    plt.plot(y_pred[:,-1], 'r-', label = 'final')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('burgers1D_init_vs_final_sol.png',format = 'png')
    plt.close()