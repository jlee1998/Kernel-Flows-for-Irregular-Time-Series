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

def Lorenz(T, dt, N_sims,s,r,b):
    N_t  = int(T//dt)
    sims = np.zeros((N_sims, N_t, 3))
    for sim in range(N_sims):
        sims[sim,0,0] = 0.
        sims[sim,0,1] = 1.
        sims[sim,0,2] = 1.05
        for i in range(1,N_t):
            sims[sim, i, 0] = sims[sim,i-1,0] + dt*(s*(sims[sim,i-1,1]-sims[sim,i-1,0]))
            sims[sim, i, 1] = sims[sim,i-1,1] + dt*(r*(sims[sim,i-1,0])-sims[sim,i-1,1] - (sims[sim,i-1,0]*sims[sim,i-1,2]))
            sims[sim, i, 2] = sims[sim,i-1,2] + dt*( -b*(sims[sim,i-1,2]) + (sims[sim,i-1,0]*sims[sim,i-1,1]))

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

def prepare_data_fast(data_x, delay, normalize, irregular_delays = None):
    data = data_x.T / normalize
    Y = data[delay:]
    
    #irregular_delays_norm = (irregular_delays-irregular_delays.mean()) / irregular_delays.std()
    if irregular_delays is not None:
        data = np.concatenate((data,irregular_delays[...,None]),-1)
    
    X = np.zeros((data.shape[0]-delay,delay*data.shape[1]))

    for i in range(X.shape[0]):
        X[i] = data[i:(i+delay)].reshape(-1)
    
    return X, Y

def prepare_irregular_data(data_x,delay, normalize, delays):

    assert delay==1 # This is currrently validated for delay 1 only. Extra stuff required to generalize.

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

    X = np.concatenate((X,delays[...,None][:-1]),-1)

    return X, Y
