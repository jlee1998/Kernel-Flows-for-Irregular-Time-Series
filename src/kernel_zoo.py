import torch
import numpy as np
#%% Kernel operations

# Returns the norm of the pairwise difference
def norm_matrix(matrix_1, matrix_2):
    norm_square_1 = torch.sum(torch.square(matrix_1), axis = 1)
    norm_square_1 = torch.reshape(norm_square_1, (-1,1))
    
    norm_square_2 = torch.sum(torch.square(matrix_2), axis = 1)
    norm_square_2 = torch.reshape(norm_square_2, (-1,1))
    
    d1=matrix_1.shape
    d2=matrix_2.shape
#    print(d1)
#    print(d2)
    if d1[1]!=d2[1]:
        matrix_1=torch.transpose(matrix_1)
    
    inner_matrix = torch.matmul(matrix_1, torch.transpose(matrix_2,0,1))
    
    norm_diff = -2 * inner_matrix + norm_square_1 + torch.transpose(norm_square_2,0,1)
#    print(norm_diff.shape)
    
    return norm_diff



# Returns the pairwise inner product
def inner_matrix(matrix_1, matrix_2):
    d1=matrix_1.shape
    d2=matrix_2.shape
    # print(d1)
    # print(d2)
    if d1[1]!=d2[1]:
        matrix_1=torch.transpose(matrix_1,0,1)
    return torch.matmul(matrix_1, torch.transpose(matrix_2,0,1))

def kernel_anl3(matrix_1, matrix_2, parameters):
    i=0
    
    matrix = norm_matrix(matrix_1, matrix_2)
    sigma = parameters[i+0]
    K =  torch.exp(-matrix/ (2* sigma**2))
    K=K*(parameters[i+1])**2
    i=i+2
    
    
    c = (parameters[i])**2
    imatrix = inner_matrix(matrix_1, matrix_2)
    K = K+ (parameters[i+1])**2 *(imatrix+c) ** 2
    i=i+2
    
    beta = parameters[i]
    gamma = (parameters[i+1])**2
    K=K+ (parameters[i+2])**2 *(beta**2 + gamma*matrix)**(-1/2)
    i=i+3
    
    alpha = parameters[i]
    beta = parameters[i+1]
    K=K+ (parameters[i+2])**2 *(beta**2 + matrix)**(-alpha)
    i=i+3
    
    sigma = parameters[i]
    K=K+ (parameters[i+1])**2 * 1/(1 + matrix/sigma**2)
    i=i+2
    
    alpha_0 = parameters[i]
    sigma_0 = parameters[i+1]
    alpha_1 = parameters[i+2]
    sigma_1 = parameters[i+3]
    K =  K+ (parameters[i+4])**2 *alpha_0*torch.maximum(torch.zeros(1,device = parameters.device), 1-matrix/(sigma_0))+ alpha_1 * torch.exp(-matrix/ (2* sigma_1**2))
    i=i+5
    
    p = parameters[i]
    l = parameters[i+1]
    sigma = parameters[i+2]
    K =K+ (parameters[i+3])**2 * torch.exp(-torch.sin(matrix*np.pi/p)**2/l**2)*torch.exp(-matrix/sigma**2)
    i=i+4
    
    p = parameters[i]
    l = parameters[i+1]
    K = K+ (parameters[i+2])**2 *torch.exp(-torch.sin(matrix*np.pi/p)/l**2)
    i=i+3
    
    return K
