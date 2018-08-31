from kuramoto import kuramoto, order_param, CRstim, rho, MNflash, rhoMN, P, rk4, mean_max_R
from math import pi
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import timeit

"""Run Kuramoto model with ON-OFF CR stim and compute mean-max of the order parameter. 
Each set of parameters takes on the order of minutes to integrate. In practice, (cf. Figs 5 and 6)
this was massively parallelized. (cf. Figs 5 and 6)
"""

# Model parameters
N = 200                                          #number of oscillators in system
K = 0.1                                          #coupling strength
mean_w = pi                                      #mean of distribution of natural frequencies of oscillators
std_w = 0.02                                     #standard deviation of disribution of natural freqs
w = np.random.normal(mean_w, std_w, N)           #natural frequencies normally distributed

#CR parameters
I = 10                                           #stim intensity
sigma = 0.4                                      #stim spatial decay rate parameter
L = 10.                                          #length of segment on which oscillators are distributed
x = np.linspace(0,N-1,N) * L/(N-1)               #oscillator locations (Nx1)
N_c = 4                                          #number of stim sites
c = np.linspace(0.5, N_c-0.5, N_c) * L/N_c       #stim locations (1xN_c)
D = 1/(1+(x.reshape(N,1)-c)**2/sigma**2)         #spatial profile value for all oscillator/stim location pairs 
T_p = 1./20                                      #inter-pulse interval
Ts = [2*pi/mean_w]                               #stimulation cycle length (will loop over this)

#ON/OFF parameters
paradigm = 'perflash'                             #can also be 'lysyansky' to replicate ON-OFF variant from Lysyansky et al. 2011
ms = [3.]                                         #number ON cycles (will loop over this)
ns = [2.]                                         #number OFF cycles (will loop over this)

# Integration parameters
theta0 = np.linspace(0, 2*pi, N, endpoint=False) #initial phases of oscillators
N_rp = 20                                        #number of rest periods
t0 = 0.                                          #integration start time
tf = None #defined inside the loop               #integration end time
tstep = 0.001                                    #time step


for Ti in range(len(Ts)):
    for mi in range(len(ms)):
        for ni in range(len(ns)):
            m = ms[mi]
            n = ns[ni]
            T = Ts[Ti]
            Tmn = (m+n)*T #for convenience
            Tm = m*T #for convenience
            tf = N_rp*Tmn
            
            # Define ODE system            
            if paradigm == 'perflash':
                r = lambda t:rho(t,N_c,T)*MNflash(t,Tmn,Tm)
            elif paradigm == 'lysyansky':
                r = lambda t:rhoMN(t, N_c, m, n, T, Tmn, Tm)
            theta_dot = lambda t, theta: kuramoto(theta, w, N, K) + CRstim(t, theta, I, D, r, lambda t:P(t,T_p))
            
            # Allocate memory
            t = np.arange(t0, tf, tstep)
            theta = np.full((t.size, N), np.nan) #each row is time step  
            
            # Run the solver
            theta[0,:] = theta0
            print('Starting T={}, m={}, n={}'.format(T,m,n))
            timeitstart = timeit.default_timer()        
            for i in range(len(t)-1):
                theta[i+1,:] = rk4(t[i], tstep, theta[i,:], theta_dot)
            print('Time to integrate: {} sec'.format(timeit.default_timer()-timeitstart))
            timeitstart = timeit.default_timer() 
            R = order_param(theta,N,k=1)[0]
            R1mm = mean_max_R(R, N_rp, T, m, n, N_rp/2, t)
            print('Time to compute order param: {} sec'.format(timeit.default_timer()-timeitstart))
            print('Mean-max of R1 = {}'.format(R1mm))
            

                
                
            
            
            
            


            
            
            
            
            
            
            
            
            
            
            
            