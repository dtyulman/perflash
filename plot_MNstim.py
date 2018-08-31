import matplotlib.pyplot as plt
import numpy as np
from kuramoto import rho, MNflash, rhoMN, rhoPerm


"""Visualize various stimuation paradigms (cf. Fig 2)
"""

#%% Parameters
mean_w = np.pi
T_w = 2*np.pi/mean_w

T = T_w 
m = 2.25
n = 1.5
Tmn = (m+n)*T
Tm = m*T
N_c = 4

tstep=0.001
t0 = 0.
tf = (m+n)*T*2
t = np.arange(t0, tf+tstep, tstep)

#%% Create axes
nplots = 5

fig, ax = plt.subplots(nrows=nplots, sharex='col')
for a in range(len(ax)):
    ax[a].set_yticks([])
    for k in range(int(np.ceil(tf/T))):
        ax[a].axvline(k*T, linestyle=':', color='k', linewidth=1)

#%% Generate plots
a = -1

a+=1
r = rho(t, N_c, T)
fl = MNflash(t, Tmn, Tm)
for i in range(N_c):
    ax[a].fill_between(t, fl*r[:,i]+i, np.zeros(t.shape)+i)
ax[a].set_ylabel('Periodic flashing')

a+=1
rhPerm = np.full((t.size, N_c), np.nan)
fl = MNflash(t, Tmn, Tm)
for ti in range(len(t)):
    rhPerm[ti,:] = rhoPerm(t[ti], N_c, T, T)
for i in range(N_c):
    ax[a].fill_between(t, fl*rhPerm[:,i]+i, np.zeros(t.shape)+i)
ax[a].set_ylabel('Periodic flashing\nrandomized')


a+=1
rMN = rhoMN(t,N_c,m,n,T,Tmn,Tm)
for i in range(N_c):
    ax[a].fill_between(t, rMN[:,i]+i, np.zeros(t.shape)+i)
ax[a].set_ylabel('Lysyansky')


a+=1
rh = rho(t, N_c, T)
for i in range(N_c):
    ax[a].fill_between(t, rh[:,i]+i, np.zeros(t.shape)+i)
ax[a].set_ylabel('Chronic CR')


a+=1
ax[a].plot(t, np.mod(mean_w*t, 2*np.pi))
ax[a].plot(t, np.zeros(t.shape), linestyle='-', color='k', linewidth=0.5)
ax[a].set_ylabel('$\Omega t$ (mod 2$\pi$)')


ax[a].set_xlabel('Time')


       













    

