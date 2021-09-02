import numpy as np
import torch
import kernel_flow

def Henon(T, dt, N_sims,a,b):
    N_t  = int(T//dt)
    sims = np.zeros((N_sims, N_t, 2))
    for i in range(1,N_t):
        sims[:, i] = np.array([1-a*sims[:,i-1,0]**2+sims[:,i-1,1],b*sims[:,i-1,0]]).T
    return sims.astype(np.float32)

# generate dataset
Data = Henon(T=200, dt=0.1,N_sims=1,a=1.4,b=0.3)[0]
#train test split
observed_data= Data[0::2]
train_data = Data[:600,:].T
test_data = Data[600:,:].T
#times
Times = np.linspace(1,2000-1,2000-1)

lenX=len(train_data[0,:])
num_modes = train_data.shape[0]

# Some constants
nparameters=24
delay = 2
regu_lambda = 1000
noptsteps = 5000
vdelay=delay*np.ones((num_modes,), dtype=int)
vregu_lambda=regu_lambda*np.ones((num_modes,))

# Get scaling factor    
normalize=np.amax(train_data[:,:])

# Prepare training data
X=np.zeros((1+lenX-2*delay,delay*num_modes))
Y=np.zeros((1+lenX-2*delay,delay*num_modes))
for mode in range(train_data.shape[0]):
    for i in range(1+lenX-2*delay):
          X[i,(mode*delay):(mode*delay+delay)]=train_data[mode,i:(i+delay)]
          Y[i,(mode*delay):(mode*delay+delay)]=train_data[mode,(i+delay):(i+2*delay)]

# Normalize
X=X/normalize
Y=Y/normalize

model = kernel_flow.KernelFlows("anl3",nparameters= 24, regu_lambda=1000)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

model.set_training_data(torch.Tensor(X),torch.Tensor(Y))

for i in range(100):
    optimizer.zero_grad()
    rho = model()
    rho.backward()
    optimizer.step()
    print(rho)

print(model.kernel_params)

