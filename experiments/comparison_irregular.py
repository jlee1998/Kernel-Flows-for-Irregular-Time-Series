import numpy as np
import torch
from kernel_flows import kernel_flow, datagen
import tqdm
import matplotlib.pyplot as plt
import time

def generate_Lorenz_data(alpha):
    max_delay = alpha
    N_points = 10000
    train_n = 5000
    burnin = 200
    dt = 0.01

    delays = np.random.randint(max_delay,size=N_points-1)+1
    indices = np.concatenate((np.zeros(1),np.cumsum(delays))).astype(int)
    delays = np.concatenate((delays,np.zeros(1))).astype(int)

    max_idx = indices[-1]
    T = np.ceil(max_idx*dt)

    # generate dataset
    Data = datagen.Lorenz(T=T+burnin*dt, dt=dt,s=10,r=28,b = 10/3, N_sims=1)[0][burnin:]

    #times
    observed_data = Data[indices]

    train_data = observed_data[:train_n,:].T
    test_data = observed_data[train_n:,:].T
    delays_train = delays[:train_n]
    delays_test = delays[train_n:]

    return train_data, test_data, delays_train, delays_test

def generate_Henon_data(alpha):
    max_delay = alpha
    N_points = 1000
    train_n = 600
    burnin = 200
    dt = 0.1

    delays = np.random.randint(max_delay,size=N_points-1)+1
    indices = np.concatenate((np.zeros(1),np.cumsum(delays))).astype(int)
    delays = np.concatenate((delays,np.zeros(1))).astype(int)

    max_idx = indices[-1]
    T = np.ceil(max_idx*dt)
    # generate dataset
    Data = datagen.Henon(T=T+burnin*dt, dt=dt,N_sims=1,a=1.4,b=0.3)[0][burnin:]

    observed_data = Data[indices]

    train_data = observed_data[:train_n,:].T
    test_data = observed_data[train_n:,:].T

    delays_train = delays[:train_n]
    delays_test = delays[train_n:]
    return train_data, test_data, delays_train, delays_test

def generate_VDP_data(alpha):
    max_delay = alpha
    N_points = 10000
    train_n = 5000
    burnin = 200
    dt = 0.0025
    sigma = 0.

    delays = np.random.randint(max_delay,size=N_points-1)+1
    indices = np.concatenate((np.zeros(1),np.cumsum(delays))).astype(int)
    delays = np.concatenate((delays,np.zeros(1))).astype(int)

    max_idx = indices[-1]
    T = np.ceil(max_idx*dt)
    # generate dataset
    #Data = datagen.Henon(T=T+burnin*dt, dt=dt,N_sims=1,a=1.4,b=0.3)[0][burnin:]

    # generate dataset
    Data = datagen.VDP(T=T+burnin*dt, dt=0.0025,sigma=sigma,rho=0,N_sims=1)[0]

    #times
    observed_data = Data[indices]

    train_data = observed_data[:train_n,:].T
    test_data = observed_data[train_n:,:].T
    delays_train = delays[:train_n]
    delays_test = delays[train_n:]

    return train_data, test_data, delays_train, delays_test

def original_kernel_flow(train_data, test_data, n_parameters, delay, regu_lambda, lr, horizon, dim, epochs = 1000):

    # Get scaling factor
    normalize=np.amax(train_data[:,:])

    X_train, Y_train = datagen.prepare_data_fast(train_data,delay,normalize)
    X_test, Y_test = datagen.prepare_data_fast(test_data,delay,normalize)

    model = kernel_flow.KernelFlows("anl3", nparameters = n_parameters, regu_lambda = regu_lambda, dim = dim, metric = "rho_ratio", batch_size = 100)

    model, rho_list  = kernel_flow.train_kernel(X_train, Y_train, model, lr = lr, epochs = epochs)

    model.compute_kernel_and_inverse(regu_lambda = regu_lambda)

    Y_pred = model.predict_ahead(X_test,horizon=horizon, delay = delay, delta_t_mode = False)

    mse_pred = (Y_pred.detach()-Y_test).pow(2).mean()
    r2 = 1-mse_pred/Y_test.var()

    return mse_pred, r2

def irregular_kernel_flow(train_data, test_data, delays_train, delays_test, n_parameters, delay, regu_lambda, lr, horizon, dim, epochs = 1000):
    # Get scaling factor
    normalize=np.amax(train_data[:,:])

    X_train, Y_train = datagen.prepare_data_fast(train_data,delay,normalize, delays_train)
    X_test, Y_test = datagen.prepare_data_fast(test_data,delay,normalize, delays_test)

    model = kernel_flow.KernelFlows("anl3", nparameters = n_parameters, regu_lambda = regu_lambda, dim = dim, metric = "rho_ratio", batch_size = 100)

    model, rho_list  = kernel_flow.train_kernel(X_train, Y_train, model, lr = lr, epochs = epochs)
    model.compute_kernel_and_inverse(regu_lambda = regu_lambda)
    Y_pred = model.predict_ahead(X_test,horizon=horizon, delay = delay, delta_t_mode = True)

    mse_pred = (Y_pred.detach()-Y_test).pow(2).mean()
    r2 = 1-mse_pred/Y_test.var()

    return mse_pred, r2

def derivative_kernel_flow(train_data, test_data, delays_train, delays_test, n_parameters, delay, regu_lambda, lr, horizon, dim, epochs = 1000):
    # Get scaling factor
    normalize=np.amax(train_data[:,:])

    X_train, Y_train = datagen.prepare_data_fast(train_data,delay,normalize, derivative_learning = True, irregular_delays = delays_train)
    X_test, Y_test_ = datagen.prepare_data_fast(test_data,delay,normalize, derivative_learning = True, irregular_delays = delays_test)
    _, Y_test = datagen.prepare_data_fast(test_data,delay,normalize)
    model = kernel_flow.KernelFlows("anl3", nparameters = n_parameters, regu_lambda = regu_lambda, dim = dim, metric = "rho_ratio", batch_size = 100)
 
    model, rho_list  = kernel_flow.train_kernel(X_train, Y_train, model, lr = lr, epochs = epochs)
    model.compute_kernel_and_inverse(regu_lambda = regu_lambda)
    #Y_pred = model.predict_ahead(X_test,horizon=horizon, delay = delay, delta_t_mode = True)
    Y_pred = model.predict_ahead(X_test,horizon=horizon, delay = delay, delta_t_mode = False, derivative_learning = True, delta_t = delays_test[:-delay])

    mse_pred = (Y_pred.detach()-Y_test).pow(2).mean()
    r2 = 1-mse_pred/Y_test.var()

    return mse_pred, r2



def LorenzExp(learning = True,alpha = 5, exp_flags = {"o":True,"i":True,"d":True}, horizon = 20, epochs = 1000, lr = 0.01):
    num_repeats = 5
    train_data, test_data, delays_train, delays_test = generate_Lorenz_data(alpha)

    if learning is False:
        epochs = 0
    else:
        epochs = epochs

    regu_lambda = 0.00001
    horizon = horizon
    delay = 2
    lr = lr
    dim = 3
    
    start_time = time.time()
    run_all_exps(train_data, test_data, delays_train, delays_test, regu_lambda, horizon, delay, lr, dim, num_repeats, epochs, exp_flags = exp_flags)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Computation time: {elapsed_time/num_repeats}")
    
    
def run_all_exps(train_data, test_data, delays_train, delays_test, regu_lambda, horizon, delay, lr, dim, num_repeats, epochs , exp_flags = {"o":True,"i":True,"d":True}):
    mse_o = []
    r2_o = []
    mse_i = []
    r2_i = []
    mse_d = []
    r2_d = []
    for rep in range(num_repeats):
        if exp_flags["o"]:
            mse_pred_o, r2_pred_o = original_kernel_flow(train_data,test_data,n_parameters=24, delay = delay, regu_lambda = regu_lambda, lr = lr, horizon = horizon, dim = dim, epochs = epochs)
            mse_o.append(mse_pred_o)
            r2_o.append(r2_pred_o)
        if exp_flags["i"]:
            mse_pred_i, r2_pred_i = irregular_kernel_flow(train_data,test_data, delays_train, delays_test, n_parameters=24, delay = delay, regu_lambda = regu_lambda, lr =lr, horizon = horizon, dim = dim, epochs = epochs)
            r2_i.append(r2_pred_i)
            mse_i.append(mse_pred_i)
        if exp_flags["d"]:
            mse_pred_d, r2_pred_d = derivative_kernel_flow(train_data,test_data, delays_train, delays_test, n_parameters=24, delay = delay, regu_lambda = regu_lambda, lr = lr, horizon = horizon, dim = dim, epochs = epochs)
            mse_d.append(mse_pred_d)
            r2_d.append(r2_pred_d)
   
    mse_o = np.array(mse_o)
    mse_i = np.array(mse_i)
    mse_d = np.array(mse_d)
    r2_o  = np.array(r2_o)
    r2_i  = np.array(r2_i)
    r2_d  = np.array(r2_d)
    
    print(f"Original : {np.nanmean(mse_o):.4f} +- {np.nanstd(mse_o):.4f} -- R2 {np.nanmean(r2_o):.4f} +- {np.nanstd(r2_o):.4f}")
    print(f"Irregular : {np.nanmean(mse_i):.4f} +- {np.nanstd(mse_i):.4f} -- R2 {np.nanmean(r2_i):.4f} +- {np.nanstd(r2_i):.4f}")
    print(f"Derivative : {np.nanmean(mse_d):.4f} +- {np.nanstd(mse_d):.4f} -- R2 {np.nanmean(r2_d):.4f} +- {np.nanstd(r2_d):.4f}")

def HenonExp(learning = True,alpha = 3, exp_flags ={"o":True,"i":True,"d":True} ):
    num_repeats = 5
    train_data, test_data, delays_train, delays_test = generate_Henon_data(alpha)

    if learning is False:
        epochs = 0
    else:
        epochs = 1000

    regu_lambda = 0.00001
    horizon = 5
    delay = 1
    lr = 0.1
    dim = 2
    
    run_all_exps(train_data, test_data, delays_train, delays_test, regu_lambda, horizon, delay, lr, dim, num_repeats, epochs, exp_flags = exp_flags)


def VDPExp(learning = True,alpha = 5, exp_flags ={"o":True,"i":True,"d":True} ):
    num_repeats = 5
    train_data, test_data, delays_train, delays_test = generate_VDP_data(alpha)

    if learning is False:
        epochs = 0
    else:
        epochs = 1000

    regu_lambda = 0.00001
    horizon = 10
    delay = 1
    lr = 0.01
    dim = 2
    
    run_all_exps(train_data, test_data, delays_train, delays_test, regu_lambda, horizon, delay, lr, dim, num_repeats, epochs, exp_flags  = exp_flags)
    
if __name__=="__main__":
  
    alpha = 5
    horizon = 20
    epochs = 1000
    #HenonExp(learning = True,alpha = 4, exp_flags = {"o":False,"i":True,"d":False})
    for lr in [0.001]:#,30,40,50]:
        print(lr)
        LorenzExp(learning = True, alpha = alpha, exp_flags = {"o":False,"i":True,"d":False}, horizon = horizon, epochs = epochs, lr = lr )
        #VDPExp(learning = True, alpha = alpha, exp_flags = {"o":True,"i":False,"d":False} ) 
