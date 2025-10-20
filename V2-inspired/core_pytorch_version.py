###############################################################################
# Import required packages
import os
import numpy as np
import time
from scipy.linalg import block_diag
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
###############################################################################

###############################################################################
# Function that predicts stochastic competitive Lotka-Volterra dynamics with
# coexistence equilibrium
# Note that "states" input can have shape (nx+nw, 1) or (nx+nw,)
def stoch_dyn_LVE(states):
    
    # Distribute states
    xk = states[0]
    yk = states[1]
    xkw = states[2]
    ykw = states[3]
    
    # Enter parameters
    k1 = 0.4
    k2 = 0.5
    xeq = 0.75
    yeq = 0.625
    d1 = 0.5
    d2 = 0.5
    dt = 0.01
    
    # Get drift coefficients
    g1x = xk*(1 - xk - k1*yk)
    g1y = yk*(1 - yk - k2*xk)
    
    # Get diffusion coefficients
    g2x = 1/2*(d1*xk*(yk-yeq))**2
    g2y = 1/2*(d2*yk*(xk-xeq))**2
    
    # Predict forward dynamics
    xkp1 = xk + g1x*dt + np.sqrt(2*g2x*dt)*xkw
    ykp1 = yk + g1y*dt + np.sqrt(2*g2y*dt)*ykw
    
    return [np.asarray([[xkp1], [ykp1]]), np.asarray([[g1x], [g1y]]), 
            np.asarray([[g2x], [g2y]])]
###############################################################################

###############################################################################
# Function that creates Unscented Transform (UT) scaling parameters and weights
def get_weights(nx, nw):
    
    # Enter UT parameters (standard)
    beta = 0
    alpha = 1
    kappa = 0
    
    # Get total system size
    n = nx + nw
    
    # Get lambda
    lam = alpha**2*(n+kappa)-n
    
    # Get weights
    Wm = np.zeros((2*n+1,1)) # mean weight vector
    Wc = np.zeros((2*n+1,1)) # covariance weight vector
    Wm[0] =lam/(n+lam)
    Wc[0] = lam/(n+lam) + ((1-alpha**2+beta))
    Wm[1:2*n+1,:] = 1/(2*(n+lam))*np.ones((2*n,1))
    Wc[1:2*n+1,:] = 1/(2*(n+lam))*np.ones((2*n,1))
    
    # Normalize weights
    Wm = Wm/np.sum(Wm)
    Wc = Wc/np.sum(Wc)
    
    # Ensure float 32 data type
    return np.float32(lam), np.float32(Wm), np.float32(Wc)
###############################################################################

###############################################################################
# Function that creates sigma point matrix for UT
# Note that we create sigma point matrices whose columns are in the following
# order: [x, u, w]^T
# Note that the mean input needs to be of shape (nx+nu, 1) and the variance
# input needs to be of shape (nx, nx)
def get_sigma(mean, variance, nx, nu, nw, lam):
    
    # Get total system size
    n = nx + nw
    
    # Combine variances of states and process noise variables into n x n matrix
    SS = block_diag(variance, np.identity(nw))
    
    # Take square root of this matrix (multiplied by (n+lam))
    S = sqrtm((n+lam)*SS)
    
    # Conatenate state mean with means of process noise variables
    concat_mean = np.concatenate((mean[0:nx,:], np.zeros((nw,1))), axis=0)
    
    # Initialize matrix of sigma points with mean values for every entry
    chi = concat_mean*np.ones((n, 2*n+1))
    
    # Adjust sigma points based on variance
    count = 0
    for i in range(1, n+1):
        chi[:,i] = chi[:,i] + S[:,count]
        count = count + 1
    
    count = 0
    for j in range(n+1,2*n+1):
        chi[:,j] = chi[:,j] - S[:,count]
        count = count + 1
    
    # Insert row(s) for exogenous input(s) (if exogenous inputs exist)
    if nu > 0:
        for i in range(0, nu):
            chi = np.insert(chi, nx+i, mean[nx+i,0]*np.ones(2*n+1), 0)
    
    return np.float32(chi)
###############################################################################

###############################################################################
# Function that performs (entire) UT propagation
# Note that the mean input needs to be of shape (nx+nu, 1) and the variance
# input needs to be of shape (nx, nx)
def UT(stoch_dyn, mean, variance, nx, nu, nw):
    
    # Get total system size
    n = nx+nw
    
    # Get weights and scaling parameters
    lam, Wm, Wc =  get_weights(nx, nw)
    
    # Get sigma points
    chi = get_sigma(mean, variance, nx, nu, nw, lam)
    
    # Propagate sigma points through stochastic dynamics
    y = np.zeros((nx, 2*n+1))
    for i in range(0, 2*n+1):
        y[:,i], _, _ = stoch_dyn(chi[:,i])
    
    # Get predicted mean and variance
    mean = np.matmul(y, Wm)
    
    variance = np.matmul(Wc.T*(y-mean), np.transpose((y-mean)))
    
    return mean, variance, y, chi
###############################################################################

###############################################################################
# Function that standardizes data
def standardize(X, mu, std):
    return (X-mu)/std

# Function that "un-standardizes" data
def un_standardize(X, mu, std):
    return (X*std)+mu

# Function that finds mean and standard deviation of data set
def find_mu_std(X):
    
    # Pre-allocate
    std = np.zeros((np.shape(X)[1]))
    mu = np.zeros((np.shape(X)[1]))
    
    # Find mu, std
    for i in range(0, np.shape(X)[1]):
        std[i] = np.std(X[:,i])
        mu[i] = np.mean(X[:,i])
    
    return mu, std
###############################################################################
    
###############################################################################
# Function that splits data into training, validation, and testing sets
def train_val_test_split(X, Y):
    
    TF = 0.60 # fraction of data used for training
    
    X_train, X_val_test, Y_train, Y_val_test  = train_test_split(X, Y, 
                                                                 test_size=(1-TF))
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test,
                                                    Y_val_test, test_size=0.50)
    
    return [X_train, X_val, X_test, Y_train, Y_val, Y_test]
###############################################################################

###############################################################################
# Function that preps training data for estimating drift coefficient as a
# neural network
# Note that the function uses UT for uncertainty propagation
# Note that mean_initial/mean_final must have shape(nx+nu, 1) while
# cov_initial/cov_final must have shape (nx, nx)
def g1_train_prep(mean_initial, mean_final, cov_initial, nx, nu, nw, lam, dt, 
                  g1_path):
    
    # Get total system size
    n = nx + nw
    
    # Initialize
    X = [] # [x(k), u(k)] where k is discrete time point (if u exists)
    Y = [] # ((X(k+1))-X(k))/dt
    CHI = [] # Sigma points

    for i in range(0, np.shape(mean_initial)[0]):
        
        # Get initial and final mean states
        xk = mean_initial[i,:,0]
        xkp1 = mean_final[i,:,0]
        
        # Get sigma points
        chi = get_sigma(mean_initial[i,:,:], cov_initial[i,:,:], nx, nu, nw, lam)
        
        # Record initial states
        X.append(np.array(xk))
    
        # Record final states
        Y.append((xkp1-xk[0:nx])/dt)
        
        # Record relevant sigma points (except for the mean)
        CHI.append(chi[0:nx+nu,1:].transpose().flatten())
    
    # Convert to array
    X = np.asarray(X)
    Y = np.asarray(Y)
    CHI = np.asarray(CHI)
    
    # Concatenate sigma points to Y so that they are easier to deal with
    # when training neural network with keras
    Y_concat = np.concatenate((Y, CHI), axis=1)
    
    # Split into training, validation, and testing data
    [X_train, X_val, X_test, 
     Y_train, Y_val, Y_test] = train_val_test_split(X, Y_concat)
    
    # Find mean and standard deviation of neural network input based on 
    # training data
    X_mu, X_std = find_mu_std(X_train)
    
    # Standardize neural network input
    X_train_scaled = standardize(X_train, X_mu, X_std)
    X_val_scaled = standardize(X_val, X_mu, X_std)
    X_test_scaled = standardize(X_test, X_mu, X_std)
    
    # Find min and max of neural network output, which
    # is the output of the drift coefficient (i.e., Y[:,0:nx])
    Y_mu, Y_std = find_mu_std(Y_train[:,0:nx])
    
    # Pre-allocate
    Y_train_scaled = np.zeros(np.shape(Y_train))
    Y_val_scaled = np.zeros(np.shape(Y_val))
    Y_test_scaled = np.zeros(np.shape(Y_test))

    # Standardize Y[:,0:nx]
    Y_train_scaled[:,0:nx] = standardize(Y_train[:,0:nx], Y_mu, Y_std)
    Y_val_scaled[:,0:nx] = standardize(Y_val[:,0:nx], Y_mu, Y_std)
    Y_test_scaled[:,0:nx] = standardize(Y_test[:,0:nx], Y_mu, Y_std)
    
    
    # Normalize sigma points (i.e., remaining columns of Y)
    for i in range(0, nx+nu):
        for j in range(0, 2*n):
            Y_train_scaled[:,nx+i+j*(nx+nu)] = standardize(Y_train[:,nx+i+j*(nx+nu)],
                                                      X_mu[i], X_std[i])
            Y_val_scaled[:,nx+i+j*(nx+nu)] = standardize(Y_val[:,nx+i+j*(nx+nu)], 
                                                    X_mu[i], X_std[i])
            Y_test_scaled[:,nx+i+j*(nx+nu)] = standardize(Y_test[:,nx+i+j*(nx+nu)], 
                                                     X_mu[i], X_std[i])
        
    
    # Ensure float32 data type and save
    X_train_scaled = np.float32(X_train_scaled)
    X_val_scaled = np.float32(X_val_scaled)
    X_test_scaled = np.float32(X_test_scaled)
    Y_train_scaled = np.float32(Y_train_scaled)
    Y_val_scaled = np.float32(Y_val_scaled)
    Y_test_scaled = np.float32(Y_test_scaled)
    X_mu = np.float32(X_mu)
    X_std = np.float32(X_std)
    Y_mu = np.float32(Y_mu)
    Y_std = np.float32(Y_std)

    np.save(os.path.join(g1_path, 'X_train_g1.npy'), X_train_scaled)
    np.save(os.path.join(g1_path, 'X_val_g1.npy'), X_val_scaled)
    np.save(os.path.join(g1_path, 'X_test_g1.npy'), X_test_scaled)
    np.save(os.path.join(g1_path, 'Y_train_g1.npy'), Y_train_scaled)
    np.save(os.path.join(g1_path, 'Y_val_g1.npy'), Y_val_scaled)
    np.save(os.path.join(g1_path, 'Y_test_g1.npy'), Y_test_scaled)
    np.save(os.path.join(g1_path, 'X_mu_g1.npy'), X_mu)
    np.save(os.path.join(g1_path, 'X_std_g1.npy'), X_std)
    np.save(os.path.join(g1_path, 'Y_mu_g1.npy'), Y_mu)
    np.save(os.path.join(g1_path, 'Y_std_g1.npy'), Y_std)
    
    return [X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, 
        Y_val_scaled, Y_test_scaled, X_mu, X_std, Y_mu, Y_std]
        
###############################################################################  
  

###############################################################################
# Function that prepares PyTorch
def prepare_pytorch():
    
    # Check for GPU computing
    if torch.cuda.is_available():
        print("GPU computing is available")
        device = torch.device("cuda")
    else:
        print("GPU computing is NOT available")
        device = torch.device("cpu")
        
    # Check PyTorch version
    print("PyTorch version: {}".format(torch.__version__))
    return device
###############################################################################

# Las funciones stoch_dyn_CSA, stoch_dyn_LVE, stoch_dyn_SIR, get_weights, 
# get_sigma, UT, standardize, un_standardize, find_mu_std, train_val_test_split
# permanecen iguales ya que usan numpy/scipy

###############################################################################
# Function that creates neural network based on various network size parameters
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, n_hidden_nodes, n_hidden_layers):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, n_hidden_nodes))
        layers.append(nn.SiLU())  # Swish activation equivalent
        
        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(n_hidden_nodes, n_hidden_nodes))
            layers.append(nn.SiLU())
        
        # Output layer
        layers.append(nn.Linear(n_hidden_nodes, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
###############################################################################

###############################################################################
# Function that compiles neural network (PyTorch doesn't need explicit compile)
def create_optimizer(model, learning_rate=0.0001):
    return optim.Adam(model.parameters(), lr=learning_rate)
###############################################################################

###############################################################################
# Early stopping callback for PyTorch
class EarlyStopping:
    def __init__(self, patience=25, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
###############################################################################

###############################################################################
# Function that trains single neural network
def train_single_NN(model, X_train, Y_train, X_val, Y_val, loss_function, 
                   device, batch_size=32, epochs=10000):
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    Y_train_tensor = torch.FloatTensor(Y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    Y_val_tensor = torch.FloatTensor(Y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create optimizer
    optimizer = create_optimizer(model)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=25)
    
    # Training history
    train_losses = []
    val_losses = []
    
    model.train()
    
    for epoch in range(epochs):
        # Training
        epoch_train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_function(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_X.size(0)
        
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = loss_function(val_outputs, Y_val_tensor).item()
            val_losses.append(val_loss)
        
        model.train()
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return model, train_losses, val_losses
###############################################################################

###############################################################################
# Function that saves neural network
def save_NN(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model
    }, path + "NN.pth")
###############################################################################

###############################################################################
# Function that loads neural network
def load_NN(path, device):
    checkpoint = torch.load(path + "NN.pth", map_location=device)
    model = checkpoint['model_architecture']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model
###############################################################################

###############################################################################
# Loss function for estimating drift coefficient during training
class G1TrainLoss(nn.Module):
    def __init__(self, nx, nu, Wm, model):
        super(G1TrainLoss, self).__init__()
        self.nx = nx
        self.nu = nu
        self.Wm = torch.FloatTensor(Wm)
        self.model = model
    
    def forward(self, y_pred, y_true):
        # Output of neural network multiplied by first mean weight
        yp_total = self.Wm[0] * y_pred[:, :self.nx]
        
        # Propagate remaining sigma points through the dynamics
        for i in range(len(self.Wm) - 1):
            sigma_points = y_true[:, (self.nx + self.nu) * i + self.nx : 
                                  (self.nx + self.nu) * i + self.nx + self.nu + self.nx]
            yp_total += self.Wm[i + 1] * self.model(sigma_points)[:, :self.nx]
        
        # Calculate squared error
        squared_difference = torch.square(y_true[:, :self.nx] - yp_total[:, :self.nx])
        
        # Get mean of squared error
        return torch.mean(squared_difference)
###############################################################################

###############################################################################
# Loss function for estimating drift coefficient for testing
def g1_test_loss(model, y_true, x_test, nx, nu, Wm, device):
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(x_test).to(device))
        y_true_tensor = torch.FloatTensor(y_true).to(device)
        
        Wm_tensor = torch.FloatTensor(Wm).to(device)
        
        # Output of neural network multiplied by first mean weight
        yp_total = Wm_tensor[0] * y_pred[:, :nx]
        
        # Propagate remaining sigma points through the dynamics
        for i in range(len(Wm) - 1):
            sigma_points = y_true_tensor[:, (nx + nu) * i + nx : 
                                        (nx + nu) * i + nx + nu + nx]
            yp_total += Wm_tensor[i + 1] * model(sigma_points)[:, :nx]
        
        # Calculate squared error
        squared_difference = torch.square(y_true_tensor[:, :nx] - yp_total[:, :nx])
        
        # Get mean of squared error
        return torch.mean(squared_difference).cpu().numpy()
###############################################################################

###############################################################################
# Function that trains multiple g1 neural networks and saves relevant outputs
def train_multiple_NNs_g1(X_train, X_val, X_test, Y_train, Y_val, Y_test, 
                         n_hidden_layers, n_hidden_nodes, nx, nu, Wm, g1_path, device):
    
    # Get input and output dimensions of neural network
    input_dim = nx + nu
    output_dim = nx

    # Train neural networks that represent g1
    for nhl in n_hidden_layers:
        for nhn in n_hidden_nodes:
            
            # Create path where model will be saved
            path = os.path.join(g1_path, str(nhl) + "_HL_" + str(nhn) + "_Nodes/")
            os.makedirs(path, exist_ok=True)
            
            # Create neural network
            model = NeuralNetwork(input_dim, output_dim, nhn, nhl).to(device)
            
            # Create loss function
            loss_fn = G1TrainLoss(nx, nu, Wm, model)
            
            # Train neural network
            start = time.time()
            model, train_losses, val_losses = train_single_NN(
                model, X_train, Y_train, X_val, Y_val, loss_fn, device)
            end = time.time()
            total_time = end - start
            
            # Save neural network
            save_NN(model, path)
            
            # Calculate test loss
            test_loss = g1_test_loss(model, Y_test, X_test, nx, nu, Wm, device)
            
            # Save losses
            np.save(path + "Train_Loss.npy", np.array(train_losses))
            np.save(path + "Val_Loss.npy", np.array(val_losses))
            np.save(path + "Test_Loss.npy", test_loss)
            
            # Save total run time
            np.save(path + "total_time.npy", total_time)
###############################################################################

###############################################################################
# Function that evaluates the deterministic part of the stochastic differential
# equation according to an Euler discretization
def det_eval(g1_model, states, X_mu_g1, X_std_g1, Y_mu_g1, Y_std_g1, nx, nu, dt, device):
    
    # Reshape input
    states = states.reshape(nx + nu,)
    
    # Scale input
    states_scaled = (states - X_mu_g1) / X_std_g1
    states_scaled = torch.FloatTensor(states_scaled.reshape(1, nx + nu)).to(device)
    
    # Get predictions from neural network and un-scale them
    g1_model.eval()
    with torch.no_grad():
        g1_outputs_scaled = g1_model(states_scaled).cpu().numpy()[0].reshape(nx,)
    
    g1_outputs = g1_outputs_scaled * Y_std_g1 + Y_mu_g1

    # Include previous state and time discretization
    return states[0:nx] + g1_outputs * dt
###############################################################################

###############################################################################
# Function that calculates "target" (i.e., Y) for training diffusion coefficient neural network
def g2_target_calc(g1_model, mean_initial, mean_final, cov_initial, cov_final, 
                   X_mu_g1, X_std_g1, Y_mu_g1, Y_std_g1, nx, nu, nw, lam, Wc, 
                   dt, device):
    
    # Get total system size
    n = nx + nw
     
    # Get sigma points
    chi = get_sigma(mean_initial, cov_initial, nx, nu, nw, lam)
    
    # Get propagated sigma points (assuming g2=0)
    y = np.zeros((nx, 2*n+1))
    for i in range(0, 2*n+1):
        y[:,i] = det_eval(g1_model, chi[0:nx+nu,i], X_mu_g1, X_std_g1, Y_mu_g1, 
                         Y_std_g1, nx, nu, dt, device)
    
    # Subtract final mean from y
    y_minus_mean = y-mean_final
    
    # "Zero out" entries that have non-zero g2 contributions
    for i in range(0, nw):
        y_minus_mean[i,1+nx+i] = 0
        y_minus_mean[i,1+nx+n+i] = 0
            
    # Multiply y_minus_mean by covariance weights and square.
    var_pt_1 = np.matmul(Wc.T*(y_minus_mean), np.transpose((y_minus_mean)))
    
    # Use propagated variance
    det_mean = det_eval(g1_model, mean_initial, X_mu_g1, X_std_g1, Y_mu_g1, 
                        Y_std_g1, nx, nu, dt, device)
    
    var_pt_2 = np.eye(nx)
    for i in range(0, nx):
        var_pt_2[i,i] = Wc[nx+1]*(2*(det_mean[i])**2 + 2*mean_final[i]**2 - 4*det_mean[i]*mean_final[i])
        
    # Get target
    target = np.array(np.diag(cov_final - var_pt_1 - var_pt_2))
    
    for i in range(0, nw):
       chi_w = chi[nx+nu+i, 1+nx+i]
       target[i] = target[i]/4/Wc[nx+1]/chi_w**2/dt
    
    return target

###############################################################################
# Function that preps training data for estimating diffusion coefficient as a neural network
def g2_train_prep(g1_model, mean_initial, mean_final, cov_initial, cov_final, 
                  X_mu_g1, X_std_g1, Y_mu_g1, Y_std_g1, nx, nu, nw, lam, Wc, 
                  dt, g2_path, device):
    
    # Get data that will be used to train the model
    X = [] # Z(k) where k is discrete time point
    Y = [] # Estimated diffusion coefficient at time k

    for i in range(0, np.shape(mean_initial)[0]):
        
        # Get initial and final mean states
        X.append(mean_initial[i,:,0])
        
        # Get sigma points
        Y.append(g2_target_calc(g1_model, mean_initial[i,:,:], mean_final[i,:,:], 
                                cov_initial[i,:,:], cov_final[i,:,:], X_mu_g1, 
                                X_std_g1, Y_mu_g1, Y_std_g1, nx, nu, nw, lam,
                                Wc, dt, device))
    # Convert to array
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    # Split into training, validation, and testing data
    [X_train, X_val, X_test, 
     Y_train, Y_val, Y_test] = train_val_test_split(X, Y)
     
    # Find mean and standard deviation of neural network input based on 
    # training data
    X_mu, X_std = find_mu_std(X_train)
    
    # Normalize neural network input
    X_train_scaled = standardize(X_train, X_mu, X_std)
    X_val_scaled = standardize(X_val, X_mu, X_std)
    X_test_scaled = standardize(X_test, X_mu, X_std)
    
    # Find min and max of neural network output, which
    # is the output of the drift coefficient (i.e., Y[:,0:nx])
    Y_mu, Y_std = find_mu_std(Y_train)
    
    # Normalize Y[:,0:nx]
    Y_train_scaled = standardize(Y_train, Y_mu, Y_std)
    Y_val_scaled = standardize(Y_val, Y_mu, Y_std)
    Y_test_scaled = standardize(Y_test, Y_mu, Y_std)
    
    # Ensure float32 data type and save
    X_train_scaled = np.float32(X_train_scaled)
    X_val_scaled = np.float32(X_val_scaled)
    X_test_scaled = np.float32(X_test_scaled)
    Y_train_scaled = np.float32(Y_train_scaled)
    Y_val_scaled = np.float32(Y_val_scaled)
    Y_test_scaled = np.float32(Y_test_scaled)
    X_mu = np.float32(X_mu)
    X_std = np.float32(X_std)
    Y_mu = np.float32(Y_mu)
    Y_std = np.float32(Y_std)

    # Save data
    np.save(os.path.join(g2_path, 'X_train_g2.npy'), X_train_scaled)
    np.save(os.path.join(g2_path, 'X_val_g2.npy'), X_val_scaled)
    np.save(os.path.join(g2_path, 'X_test_g2.npy'), X_test_scaled)
    np.save(os.path.join(g2_path, 'Y_train_g2.npy'), Y_train_scaled)
    np.save(os.path.join(g2_path, 'Y_val_g2.npy'), Y_val_scaled)
    np.save(os.path.join(g2_path, 'Y_test_g2.npy'), Y_test_scaled)
    np.save(os.path.join(g2_path, 'X_mu_g2.npy'), X_mu)
    np.save(os.path.join(g2_path, 'X_std_g2.npy'), X_std)
    np.save(os.path.join(g2_path, 'Y_mu_g2.npy'), Y_mu)
    np.save(os.path.join(g2_path, 'Y_std_g2.npy'), Y_std)

    return [X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, 
            Y_val_scaled, Y_test_scaled, X_mu, X_std, Y_mu, Y_std]
###############################################################################

###############################################################################
# Loss function for estimating diffusion coefficient for testing
def g2_test_loss(model, y_true, x_test, device):
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(x_test).to(device)
        y_true_tensor = torch.FloatTensor(y_true).to(device)
        
        y_pred = model(x_test_tensor)
        squared_difference = torch.square(y_true_tensor - y_pred)
        
        return torch.mean(squared_difference).cpu().numpy()
###############################################################################

###############################################################################
# Function that trains multiple neural networks and saves relevant outputs for g2
def train_multiple_NNs_g2(X_train, X_val, X_test, Y_train, Y_val, Y_test,
                         n_hidden_layers, n_hidden_nodes, nx, nu, g2_path, device):
    
    # Get input and output dimensions of neural network
    input_dim = nx + nu
    output_dim = nx

    # Train neural networks that represent g2
    for nhl in n_hidden_layers:
        for nhn in n_hidden_nodes:
            
            # Create path where model will be saved
            path = os.path.join(g2_path, str(nhl) + "_HL_" + str(nhn) + "_Nodes/")
            os.makedirs(path, exist_ok=True)
            
            # Create neural network
            model = NeuralNetwork(input_dim, output_dim, nhn, nhl).to(device)
            
            # Create loss function (MSE)
            loss_fn = nn.MSELoss()
            
            # Train neural network
            start = time.time()
            model, train_losses, val_losses = train_single_NN(
                model, X_train, Y_train, X_val, Y_val, loss_fn, device)
            end = time.time()
            total_time = end - start
            
            # Save neural network
            save_NN(model, path)
            
            # Calculate test loss
            test_loss = g2_test_loss(model, Y_test, X_test, device)
            
            # Save losses
            np.save(path + "Train_Loss.npy", np.array(train_losses))
            np.save(path + "Val_Loss.npy", np.array(val_losses))
            np.save(path + "Test_Loss.npy", test_loss)
            
            # Save total run time
            np.save(path + "total_time.npy", total_time)
###############################################################################

###############################################################################
# Hidden physics neural network function for Lotka-Volterra system.
def LVE_NN(model, states, x_mu, x_std, g_mu, g_std, device):
    model.eval()
    with torch.no_grad():
        # Pre-allocate
        states_scaled = np.zeros((1,2))
        
        # Scale states
        states_scaled[0,0] = (states[0]-x_mu[0])/(x_std[0])
        states_scaled[0,1] = (states[1]-x_mu[1])/(x_std[1])
        
        # Get NN output
        states_tensor = torch.FloatTensor(states_scaled).to(device)
        output = model(states_tensor).cpu().numpy()[0]
        output_1 = output[0]
        output_2 = output[1]
        
        # Scale output and return value
        return (output_1)*(g_std[0])+g_mu[0], (output_2)*(g_std[1])+g_mu[1]

###############################################################################
# Function that plots reconstructed hidden physics for Lotka-Volterra system       
def plot_reconstruction_LVE(model, x_mu, x_std, g_mu, g_std, hp_type, path, device):
    
    # Minimum and maximum state values
    x_min = [0,0]
    x_max = [2,2]
    
    # Choose to "plot" one dimension and "hold" another dimension.
    x1 = np.linspace(x_min[0], x_max[0], 1000) # "Plotting"
    x2 = np.linspace(x_min[1], x_max[1], 8) # "Holding"
    
    # Create plots
    for j in range(0, len(x2)):
        
        # Initialize
        g_true_list_1 = []
        g_true_list_2 = []
        
        g_pred_list_1 = []
        g_pred_list_2 = []
        
        for i in range(0, len(x1)):
            
            # Get state
            states= np.asarray([x1[i], x2[j], 0, 0])
            
            # Get true prediction
            _, g1_true, g2_true = stoch_dyn_LVE(states)
            
            if hp_type == "g1":
                g_true_list_1.append(g1_true[0])
                g_true_list_2.append(g1_true[1])
            elif hp_type == "g2":
                g_true_list_1.append(g2_true[0])
                g_true_list_2.append(g2_true[1])
      
            # Get NN prediction
            pred = LVE_NN(model, states, x_mu, x_std, g_mu, g_std, device)
            g_pred_list_1.append(pred[0])
            g_pred_list_2.append(pred[1])
        
        # Plot for first output dimension
        plt.figure(1)
        if j == 0:
            plt.plot(x1, g_pred_list_1, color="red", linewidth=2, label="SPINN")
            plt.plot(x1, g_true_list_1, color = "black", linestyle=":", linewidth=3, label="True")
        else:
            plt.plot(x1, g_pred_list_1, color="red", linewidth=2, label=None)
            plt.plot(x1, g_true_list_1, color = "black", linestyle=":", linewidth=3, label=None)
        plt.legend(fontsize=16)
        plt.xlim([x_min[0], x_max[0]])
        plt.xlabel("$x_1$", fontsize=20)
        if hp_type == "g1":
            plt.ylabel("$g_1(x_1,x_2)_1$", fontsize=20)
        elif hp_type == "g2":
            plt.ylabel("$g_2(x_1,x_2)_1$", fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(path, hp_type + "_1_LVE.png"))
        plt.close()
        
        # Plot for second output dimension
        plt.figure(2)
        if j == 0:
            plt.plot(x1, g_pred_list_2, color="red", linewidth=2, label="SPINN")
            plt.plot(x1, g_true_list_2, color = "black", linestyle=":", linewidth=3, label="True")
        else:
            plt.plot(x1, g_pred_list_2, color="red", linewidth=2, label=None)
            plt.plot(x1, g_true_list_2, color = "black", linestyle=":", linewidth=3, label=None)
        plt.legend(fontsize=16)
        plt.xlim([x_min[0], x_max[0]])
        plt.xlabel("$x_1$", fontsize=20)
        if hp_type == "g1":
            plt.ylabel("$g_1(x_1,x_2)_2$", fontsize=20)
        elif hp_type == "g2":
            plt.ylabel("$g_2(x_1,x_2)_2$", fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(path, hp_type + "_2_LVE.png"))
        plt.close()
