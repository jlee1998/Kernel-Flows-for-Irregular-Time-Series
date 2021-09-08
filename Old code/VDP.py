# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 16:00:41 2021

@author: jhlee
"""

import matplotlib.pyplot as plt


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

# generate dataset
Data = VDP(T=25, dt=0.0025,sigma=1.2,rho=0,N_sims=1)[0]
#train test split
observed_data= Data[0::2]
train_data = Data[:2000,:].T
test_data = Data[2000:,:].T
#times
Times = np.linspace(1,2000-1,2000-1)
#take random subset
plt.plot(Data)
plt.title("Trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.show()