import numpy as np
import torch
import math
from src import kernel_zoo
import tqdm

def sample_selection(N, size):
    indices = np.arange(N)
    sample_indices = np.sort(np.random.choice(indices, size, replace= False))
    #sample_indices = indices[:size]
    return sample_indices

# The pi or selection matrix
def pi_matrix(sample_indices, dimension):
    pi = torch.zeros(dimension).double()
    
    for i in range(dimension[0]):
        pi[i][sample_indices[i]] = 1
    
    return pi

def batch_creation(N, batch_size, sample_proportion = 0.5):
    if batch_size == False:
        batch_indices = np.arange(N)
    elif 0 < batch_size <= 1:
        batch_size = int(N * batch_size)
        batch_indices = sample_selection(N, batch_size)
    else:
        batch_indices = sample_selection(N, batch_size)
        
    # Sample from the mini-batch
    sample_size = math.ceil(len(batch_indices)*sample_proportion)
    sample_indices = sample_selection(len(batch_indices), sample_size)
    
    return sample_indices, batch_indices

class KernelFlowsSG(torch.nn.Module):

    def __init__(self, kernel_keyword, nparameters, regu_lambda, dim, metric = "rho_ratio", batch_size = 100, device = torch.device("cpu")):
        super().__init__()
        self.kernel_keyword = kernel_keyword
        
        self.regu_lambda = regu_lambda

        self.kernel_params = torch.nn.Parameter(torch.ones(nparameters),requires_grad = True)
        self.kernel = kernel_zoo.kernel_anl3

        self.dim = dim

        self.batch_size = batch_size
        self.device = device
        self.metric = metric

        if metric == "rho_ratio":
            self.rho_fun = self.rho_ratio
        elif metric == "rho_general":
            self.rho_fun = self.rho_general
        elif metric == "rho_ratio_k":
            self.rho_fun = self.rho_ratio_k
        else:
            raise("Metric not supported")

    def get_parameters(self):
        return self.kernel_params
    
    def set_train(self, train):
        self.train = train

    def set_training_data(self,X,Y):
        self.X_train = X
        self.Y_train = Y
        self.N = self.X_train.shape[0] # Number of samples
    
    def prepare_semi_group(self,n_z, delays_train):
        
        random_idx = np.random.choice(self.X_train.shape[0]-1,n_z, replace = False)
        random_delays_pre = delays_train[random_idx]
        random_delays_post = delays_train[random_idx+1]

        self.X_train_phi2 = torch.cat((self.X_train[random_idx,:-1],random_delays_pre[...,None]),-1)
        
        self.z_tensor = self.X_train[random_idx+1,:-1] 
        
        self.X_train_phi3_a = torch.cat((self.z_tensor,random_delays_post[...,None]),-1)
        self.X_train_phi3_b = -  torch.cat((self.X_train[random_idx,:-1],random_delays_pre[...,None]+random_delays_post[...,None]),-1)

        self.X_train = ( (self.X_train,self.X_train_phi2,self.X_train_phi3_a), (self.X_train,self.X_train_phi2,self.X_train_phi3_b))
        self.Y_train  = (self.Y_train,self.z_tensor,torch.zeros_like(self.z_tensor))

        self.Nz = n_z # Number of z samples

    def make_kernel_old(self,matrix_data1,matrix_data2,kernel_params, n_1, n_2, n_3, one_sided = False):
        
        K = 0
        if one_sided:
            weights = torch.ones((matrix_data1[0][0].shape[0],n_1+n_2+n_3))
            weights[:,:n_1+n_2] = 0.5
        else:
            weights = torch.ones((n_1+n_2+n_3,n_1+n_2+n_3))
            weights[:n_1+n_2,:n_1+n_2] = 0.5
        assert len(matrix_data1)==len(matrix_data2)
        for i in range(len(matrix_data1)):
            #import ipdb; ipdb.set_trace()
            K = K + weights * self.kernel(torch.cat(matrix_data1[i]),torch.cat(matrix_data2[i]),kernel_params)
        return K
    
    def make_kernel(self,matrix_data1,matrix_data2,kernel_params, n_1, n_2, n_3, one_sided = False):
        

        #Phi0 and 1
        if one_sided:
            N_ = matrix_data1.shape[0]
            K = torch.zeros(N_,(n_1+n_2+n_3), device = kernel_params.device).double()
            K[:,:n_1+n_2] = self.kernel(matrix_data1,torch.cat(matrix_data2[0][:2]),kernel_params)
            K[:,n_1+n_2:] = self.kernel(matrix_data1,matrix_data2[0][2],kernel_params) - self.kernel(matrix_data1,matrix_data2[1][2],kernel_params)

        else:
            K = torch.zeros((n_1+n_2+n_3),(n_1+n_2+n_3), device = kernel_params.device).double()
            K[:n_1+n_2,:n_1+n_2] = self.kernel(torch.cat(matrix_data1[0][:2]),torch.cat(matrix_data2[0][:2]),kernel_params)
            K[:n_1+n_2,n_1+n_2:] = self.kernel(torch.cat(matrix_data1[0][:2]),matrix_data2[0][2],kernel_params) - self.kernel(torch.cat(matrix_data1[0][:2]),matrix_data2[1][2],kernel_params)
            K[n_1+n_2:,:n_1+n_2] = self.kernel(matrix_data1[0][2],torch.cat(matrix_data2[0][:2]),kernel_params) - self.kernel(matrix_data1[1][2],torch.cat(matrix_data2[0][:2]),kernel_params) 
            K[n_1+n_2:,n_1+n_2:] = self.kernel(matrix_data1[0][2],matrix_data2[0][2],kernel_params) - self.kernel(matrix_data1[0][2],matrix_data2[1][2],kernel_params) - self.kernel(matrix_data1[1][2],matrix_data2[0][2],kernel_params) + self.kernel(matrix_data1[1][2],matrix_data2[1][2],kernel_params)
        
        return K
        
    
    def rho_general(self, matrix_data, Y_data,  regu_lambda = 0.000001, ns = None, **kwargs):
        
        kernel_matrix = self.make_kernel(matrix_data,matrix_data, self.kernel_params, ns[0], ns[1], ns[2])

        kernel_matrix = kernel_matrix / torch.trace(kernel_matrix)
        
        inverse_matrix = torch.linalg.inv(kernel_matrix + regu_lambda * torch.eye(kernel_matrix.shape[0]))
        
        rho = torch.tensordot(torch.cat(Y_data), torch.matmul(inverse_matrix, torch.cat(Y_data)))
        
        return rho

    def rho_ratio_k(self, matrix_data, Y_data, sample_indices, sample_indices_z,  regu_lambda = 0.000001, ns = None, k = 1, **kwargs):
        rho_ = 0
        for idx in np.array_split(sample_indices,np.ceil(len(sample_indices)/k)):
            rho_ = rho_ + self.rho_ratio( matrix_data = matrix_data, Y_data = Y_data, sample_indices = np.array([idx]), sample_indices_z = sample_indices_z,  regu_lambda = regu_lambda, ns=ns , **kwargs)
        return rho_ / len(sample_indices)

    def rho_ratio(self, matrix_data, Y_data, sample_indices, sample_indices_z,  regu_lambda = 0.000001, ns = None, **kwargs):
       
        kernel_matrix = self.make_kernel(matrix_data,matrix_data, self.kernel_params, ns[0], ns[1], ns[2])

        sample_indices_ = np.concatenate((sample_indices, sample_indices_z+self.batch_size, sample_indices_z+ self.batch_size + self.batch_size_z))
        N_ = ns[0] + ns[1] +ns[2] 
        pi = pi_matrix(sample_indices_, (sample_indices_.shape[0], N_)).to(self.device) 
    #    print(pi.shape)
        
        sample_matrix = torch.matmul(pi, torch.matmul(kernel_matrix, torch.transpose(pi,0,1)))
    #    print(sample_matrix.shape)
        
        Y_sample = torch.cat(Y_data)[sample_indices_]
    #    print(Y_sample.shape)
        
        inverse_data = torch.linalg.inv(kernel_matrix + regu_lambda * torch.eye(kernel_matrix.shape[0], device = self.device))
        inverse_sample = torch.linalg.inv(sample_matrix + regu_lambda * torch.eye(sample_matrix.shape[0], device = self.device))
    #    print(inverse_sample.shape)
    #    B=np.matmul(inverse_sample, Y_sample)
    #    print(B.shape)
        top = torch.tensordot(Y_sample, torch.matmul(inverse_sample, Y_sample))
        
        bottom = torch.tensordot(torch.cat(Y_data), torch.matmul(inverse_data, torch.cat(Y_data)))
        
        return 1 - top/bottom 

    def forward(self, adaptive_size = False, proportion = 0.5):            
        
        if adaptive_size == False or adaptive_size == "Dynamic":
            sample_size = proportion
        elif adaptive_size == "Linear":
            sample_size_array = sample_size_linear(iterations, adaptive_range) 
        else:
            print("Sample size not recognized")
            
                
        # Create a batch and a sample
        sample_indices, batch_indices = batch_creation(self.N, batch_size= self.batch_size, sample_proportion = sample_size)
        
        batch_size_z = self.Nz #min(self.batch_size,self.Nz)
        self.batch_size_z = batch_size_z
        sample_indices_z, batch_indices_z = batch_creation(self.Nz, batch_size= batch_size_z, sample_proportion = sample_size)

        X_data = tuple( (x[0][batch_indices],x[1][batch_indices_z],x[2][batch_indices_z]) for x in self.X_train)
        Y_data = (self.Y_train[0][batch_indices], self.Y_train[1][batch_indices_z], self.Y_train[2][batch_indices_z])
        
        #optimizer and backward

        rho = self.rho_fun( X_data, Y_data, regu_lambda = self.regu_lambda, ns = (len(batch_indices),batch_size_z,batch_size_z), sample_indices = sample_indices, sample_indices_z = sample_indices_z)
           
        return rho

    def compute_kernel_and_inverse(self,regu_lambda = 0.0000001):
        X_data = self.X_train

        self.kernel_matrix = self.make_kernel(X_data, X_data,self.kernel_params, self.N, self.Nz, self.Nz)
        self.kernel_matrix += regu_lambda * torch.eye(self.kernel_matrix.shape[0], device = X_data[0][0].device)
        
        self.inverse_kernel = torch.linalg.inv(self.kernel_matrix)
        self.A_matrix = torch.matmul(self.inverse_kernel,torch.cat(self.Y_train))

    def predict(self,x_test):
        
        kernel_pred = self.make_kernel(x_test,self.X_train,self.kernel_params, n_1 = self.N, n_2 = self.Nz,  n_3 = self.Nz , one_sided = True)
        prediction = torch.matmul(kernel_pred,self.A_matrix)
        return prediction

    def predict_ahead(self,x_test, horizon, delay, delta_t_mode = False, device = None):
        """
        Perform n=horizon steps ahead prediction.

        If delta_t_mode is True, x_test is expected to have the the following structure (X(t-1),delta_t-1,X(t),delta_t))
        
        out_dim is the dimension of the y vector (and of the observations in x as well)

        delay : delay used in the x
        """
        assert horizon >0 # minimum horizon is 1
        assert delay >0 

        Y_p = torch.zeros((x_test.shape[0],self.dim))
        X_test_ = torch.Tensor(x_test).double().to(device)

        if delta_t_mode:
            indices_delays = [((self.dim+1)*i,(self.dim+1)*i+1) for i in range(delay)] # We should not touch the delta t
        else:
            indices_delays = [(self.dim*i,self.dim*i+1) for i in range(delay)]
        
        # Make sure there is no contamination (deleting the previous values)
        for dim in range(horizon):
            n_delays = min(dim,delay)
            for n in range(1,n_delays+1):
                X_test_[dim::horizon][:,indices_delays[-n]] = 0

        # Predicting and propagating predictions to the next step.
        for dim in range(horizon):
    
            Y_p[dim::horizon] = self.predict(X_test_[dim::horizon])
            
            for dim_plus in range(dim+1,min(horizon,delay+dim+1)):
                
                l_x = X_test_[dim_plus::horizon].shape[0]
                idx = dim_plus-dim

                X_test_[dim_plus::horizon][:,indices_delays[-1*idx][0]:1+indices_delays[-1*idx][1]] = Y_p[dim::horizon][:l_x] # should be -1-2:-1 for irregular

        return Y_p


class KernelFlows(torch.nn.Module):
    
    def __init__(self, kernel_keyword, nparameters, regu_lambda, dim, metric = "rho_ratio", batch_size = 100):
        super().__init__()
        self.kernel_keyword = kernel_keyword
        
        self.regu_lambda = regu_lambda

        self.kernel_params = torch.nn.Parameter(torch.ones(nparameters),requires_grad = True)
        self.kernel = kernel_zoo.kernel_anl3

        self.dim = dim

        self.batch_size = batch_size

        if metric == "rho_ratio":
            self.rho_fun = self.rho_ratio
        elif metric == "rho_general":
            self.rho_fun = self.rho_general
        else:
            raise("Metric not supported")
        self.metric = metric

    def get_parameters(self):
        return self.kernel_params
    
    def set_train(self, train):
        self.train = train

    def set_training_data(self,X,Y):
        self.X_train = X
        self.Y_train = Y

    def prepare_semi_group(self,n_z, max_delay, device):
        
        random_idx = np.random.choice(self.X_train.shape[0],n_z, replace = False)
        random_delays_pre = torch.Tensor(np.random.randint(max_delay,size=n_z)+1, device = device)
        random_delays_post = torch.Tensor(np.random.randint(max_delay,size=n_z)+1, device = device)

        self.X_train_phi2 = torch.cat((self.X_train[random_idx,:-1],random_delays_pre[...,None]),-1)
        
        self.z_tensor = torch.nn.Parameter(torch.randn(n_z,self.dim),requires_grad = True)

        self.X_train_phi3 = torch.cat((self.z_tensor,random_delays_post[...,None]),-1) -  torch.cat((self.X_train[random_idx,:-1],random_delays_pre[...,None]+random_delays_post[...,None]),-1)

        self.X_train = torch.cat((self.X_train,self.X_train_phi2,self.X_train_phi3))
        self.Y_train  = torch.cat((self.Y_train,self.z_tensor,torch.zeros_like(self.z_tensor, device = device)))

    def prepare_semi_group_new(self,n_z, delays_train):
        
        random_idx = np.random.choice(self.X_train.shape[0]-1,n_z, replace = False)
        random_delays_pre = delays_train[random_idx]
        random_delays_post = delays_train[random_idx+1]

        self.X_train_phi2 = torch.cat((self.X_train[random_idx,:-1],random_delays_pre[...,None]),-1)
        
        self.z_tensor = self.X_train[random_idx+1,:-1] 
        
        self.X_train_phi3 = torch.cat((self.z_tensor,random_delays_post[...,None]),-1) -  torch.cat((self.X_train[random_idx,:-1],random_delays_pre[...,None]+random_delays_post[...,None]),-1)
        #self.X_train_phi3 = torch.cat((self.X_train[random_idx,:-1],random_delays_pre[...,None]+random_delays_post[...,None]),-1)

        self.X_train = torch.cat((self.X_train,self.X_train_phi2,self.X_train_phi3))
        self.Y_train  = torch.cat((self.Y_train,self.z_tensor,torch.zeros_like(self.z_tensor)))

    def prepare_semi_group_simple(self,n_z, delays_train):
        
        random_idx = np.random.choice(self.X_train.shape[0]-2,n_z, replace = False)
        random_delays_pre = delays_train[random_idx]
        random_delays_post = delays_train[random_idx+1]

        self.X_train_phi2 = torch.cat((self.X_train[random_idx,:-1],random_delays_pre[...,None]+random_delays_post[...,None]),-1)
        
        self.z_tensor = self.X_train[random_idx+2,:-1] 
        
        #self.X_train_phi3 = torch.cat((self.X_train[random_idx,:-1],random_delays_pre[...,None]+random_delays_post[...,None]),-1)

        self.X_train = torch.cat((self.X_train,self.X_train_phi2))
        self.Y_train  = torch.cat((self.Y_train,self.z_tensor))

    def rho_ratio(self, matrix_data, Y_data, sample_indices,  regu_lambda = 0.000001):
        
        kernel_matrix = self.kernel(matrix_data, matrix_data, self.kernel_params)
    #    print(kernel_matrix.shape)
        pi = pi_matrix(sample_indices, (sample_indices.shape[0], matrix_data.shape[0])).to(matrix_data.device) 
    #    print(pi.shape)
        
        sample_matrix = torch.matmul(pi, torch.matmul(kernel_matrix, torch.transpose(pi,0,1)))
    #    print(sample_matrix.shape)
        
        Y_sample = Y_data[sample_indices]
    #    print(Y_sample.shape)
        
        inverse_data = torch.linalg.inv(kernel_matrix + regu_lambda * torch.eye(kernel_matrix.shape[0], device = matrix_data.device))
        inverse_sample = torch.linalg.inv(sample_matrix + regu_lambda * torch.eye(sample_matrix.shape[0], device = matrix_data.device))
    #    print(inverse_sample.shape)
    #    B=np.matmul(inverse_sample, Y_sample)
    #    print(B.shape)
        top = torch.tensordot(Y_sample, torch.matmul(inverse_sample, Y_sample))
        
        bottom = torch.tensordot(Y_data, torch.matmul(inverse_data, Y_data))
        
        return 1 - top/bottom 
    
    def rho_general(self, matrix_data, Y_data,  regu_lambda = 0.000001, **kwargs):
        
        kernel_matrix = self.kernel(matrix_data, matrix_data, self.kernel_params)

        kernel_matrix = kernel_matrix / torch.trace(kernel_matrix)
        
        inverse_matrix = torch.linalg.inv(kernel_matrix + regu_lambda * torch.eye(kernel_matrix.shape[0]))
        
        rho = torch.tensordot(Y_data, torch.matmul(inverse_matrix, Y_data))
        
        return rho

    def forward(self, adaptive_size = False, proportion = 0.5):            
        
        if adaptive_size == False or adaptive_size == "Dynamic":
            sample_size = proportion
        elif adaptive_size == "Linear":
            sample_size_array = sample_size_linear(iterations, adaptive_range) 
        else:
            print("Sample size not recognized")
            
                
        # Create a batch and a sample
        sample_indices, batch_indices = batch_creation(self.X_train.shape[0], batch_size= self.batch_size, sample_proportion = sample_size)
        X_data = self.X_train[batch_indices]
        Y_data = self.Y_train[batch_indices]
        
        #optimizer and backward

        rho = self.rho_fun( X_data, Y_data, 
                                       sample_indices = sample_indices, regu_lambda = self.regu_lambda)
           
        return rho

    def compute_kernel_and_inverse(self,regu_lambda = 0.0000001):
        X_data = self.X_train
        self.kernel_matrix = self.kernel(X_data,X_data, self.kernel_params)
        self.kernel_matrix += regu_lambda * torch.eye(self.kernel_matrix.shape[0], device = X_data.device)
        
        self.inverse_kernel = torch.linalg.inv(self.kernel_matrix)
        self.A_matrix = torch.matmul(self.inverse_kernel,self.Y_train)

    def predict(self,x_test):
        kernel_pred = self.kernel(x_test,self.X_train,self.kernel_params)
        prediction = torch.matmul(kernel_pred,self.A_matrix)
        return prediction

    def predict_ahead(self,x_test, horizon, delay, delta_t_mode = False, device = None):
        """
        Perform n=horizon steps ahead prediction.

        If delta_t_mode is True, x_test is expected to have the the following structure (X(t-1),delta_t-1,X(t),delta_t))
        
        out_dim is the dimension of the y vector (and of the observations in x as well)

        delay : delay used in the x
        """
        assert horizon >0 # minimum horizon is 1
        assert delay >0 

        Y_p = torch.zeros((x_test.shape[0],self.dim))
        X_test_ = torch.Tensor(x_test).double().to(device)

        if delta_t_mode:
            indices_delays = [((self.dim+1)*i,(self.dim+1)*i+1) for i in range(delay)] # We should not touch the delta t
        else:
            indices_delays = [(self.dim*i,self.dim*i+1) for i in range(delay)]
        
        # Make sure there is no contamination (deleting the previous values)
        for dim in range(horizon):
            n_delays = min(dim,delay)
            for n in range(1,n_delays+1):
                X_test_[dim::horizon][:,indices_delays[-n]] = 0

        # Predicting and propagating predictions to the next step.
        for dim in range(horizon):
    
            Y_p[dim::horizon] = self.predict(X_test_[dim::horizon])
            
            for dim_plus in range(dim+1,min(horizon,delay+dim+1)):
                
                l_x = X_test_[dim_plus::horizon].shape[0]
                idx = dim_plus-dim

                X_test_[dim_plus::horizon][:,indices_delays[-1*idx][0]:1+indices_delays[-1*idx][1]] = Y_p[dim::horizon][:l_x] # should be -1-2:-1 for irregular

        return Y_p

def train_kernel(X_train, Y_train, model,  lr = 0.1, verbose= False):
    """
    dim is the dimension of a single observation
    """ 
    #model = KernelFlows(kernel_name,nparameters= nparameters, regu_lambda=regu_lambda, dim = dim)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)

    model.set_training_data(torch.Tensor(X_train).double(),torch.Tensor(Y_train).double())

    for i in tqdm.tqdm(range(1000)):
        optimizer.zero_grad()
        rho = model.forward()
        if rho>=0 and rho<=1 and model.metric=="rho_ratio":
            rho.backward()
            optimizer.step()
            if verbose:
                print(rho)
            
    return model
