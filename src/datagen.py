import numpy as np

def VDP(T, dt, N_sims, sigma, rho, random_theta=False):
    N_t  = int(T//dt)
    sims = np.zeros((N_sims, N_t, 2))
    cov  = dt * np.array(
            [[sigma**2,       sigma**2 * rho],
             [sigma**2 * rho, sigma**2]])
    dW   = np.random.multivariate_normal([0, 0], cov, size=(N_sims, N_t))
    for j in range(0,N_sims):
      for i in range(1,N_t):
        sims[j,i,0] = sims[j,(i-1),0] + 100*(sims[j,(i-1),1]-6.75*(sims[j,(i-1),0]+1)*sims[j,(i-1),0]**2)*dt + dW[j,i,0]
        sims[j,i,1] = sims[j,(i-1),1] + (-0.5-sims[j,(i-1),0])*dt + 0.1*dW[j,i,1]


    return sims.astype(np.float32)

def Henon(T, dt, N_sims,a,b):
    N_t  = int(T//dt)
    sims = np.zeros((N_sims, N_t, 2))
    for i in range(1,N_t):
        sims[:, i] = np.array([1-a*sims[:,i-1,0]**2+sims[:,i-1,1],b*sims[:,i-1,0]]).T
    return sims.astype(np.float32)

def prepare_data(data_x,delay, normalize):
    lenX=data_x.shape[1]
    num_modes = data_x.shape[0]
    
    X=np.zeros((1+lenX-2*delay,delay*num_modes))
    Y=np.zeros((1+lenX-2*delay,delay*num_modes))
    for mode in range(num_modes):
        for i in range(1+lenX-2*delay):
            X[i,(mode*delay):(mode*delay+delay)]=data_x[mode,i:(i+delay)]
            Y[i,(mode*delay):(mode*delay+delay)]=data_x[mode,(i+delay):(i+2*delay)]
    
            
    # Normalize
    X=X/normalize
    Y=Y/normalize
    return X, Y
