from kuramoto import kuramoto, order_param, CRstim, rho, P, rk4
from math import pi
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import timeit

"""Run the Kuramoto model with "chronic" CR stim, varying stimulation spread (sigma) and intensity (I)
Plot average order parameter as a function of I and sigma (cf. Lysyansky et al. 2011, Fig 3)
Note that this can take quite a while to run and would benefit from parallelization
"""

# Model parameters
N = 50                                           #number of oscillators in system
K = 0.1                                          #coupling strength
mean_w = pi                                      #mean of distribution of natural frequencies of oscillators
std_w = 0.02                                     #standard deviation of disribution of natural freqs
w = np.random.normal(mean_w, std_w, N)           #natural frequencies normally distributed

#Stim parameters
Is = np.arange(1,31,2, dtype=np.float64)         #stim intensity (will loop over this)
sigmas = np.arange(0.1,3,0.2, dtype=np.float64)  #stim spatial decay rate parameter (will loop over this)
L = 10.                                          #length of segment on which oscillators are distributed
x = np.linspace(0,N-1,N) * L/(N-1)               #oscillator locations (Nx1)
N_c = 4                                          #number of stim sites
c = np.linspace(0.5, N_c-0.5, N_c) * L/N_c       #stim locations (1xN_c)
T_p = 1./20                                      #inter-pulse interval
T = 2*pi/mean_w                                  #stimulation cycle length

# Integration parameters
theta0 = np.linspace(0, 2*pi, N, endpoint=False) #initial phases of oscillators
t0 = 0.                                          #integration start time
tf = 50.                                         #integration end time
tstep = 0.001                                    #time step

R1s = np.full((len(Is), len(sigmas)), np.nan)
R4s = np.full((len(Is), len(sigmas)), np.nan)

for Ii in range(len(Is)):
    for sigmai in range(len(sigmas)):
        I = Is[Ii]
        sigma = sigmas[sigmai]       
        D = 1/(1+(x.reshape(N,1)-c)**2/sigma**2) #spatial profile value for all oscillator/stim location pairs 
        
        theta_dot = lambda t, theta: kuramoto(theta, w, N, K) + CRstim(t, theta, I, D, lambda t:rho(t,N_c,T), lambda t:P(t,T_p))
        
        # Allocate memory
        t = np.arange(t0, tf, tstep)
        theta = np.full((t.size, N), np.nan) #each row is time step  
        r = np.full(t.shape, np.nan)
        
        # Run the solver
        theta[0,:] = theta0
        r[0] = order_param(theta[0,:], N)[0]         
        print('Starting (I, sigma)=({}, {})'.format(I, sigma))
        timeitstart = timeit.default_timer()        
        for i in xrange(len(t)-1):
            theta[i+1,:] = rk4(t[i], tstep, theta[i,:], theta_dot) #use my own RK4 implementation
        print('Time to integrate: {} sec'.format(timeit.default_timer()-timeitstart))
    
        # Save the average order parameter for this (I,sigma)       
        R1s[sigmai,Ii] = np.mean(order_param(theta,N,k=1)[0][int(np.ceil(len(t)/2)):])
        R4s[sigmai,Ii] = np.mean(order_param(theta,N,k=4)[0][int(np.ceil(len(t)/2)):])
        
# Plot 
R1fig, R1ax = plt.subplots()
R1ax.set_xlabel('I')
R1ax.set_ylabel('$\sigma$')
R1ax.set_title('$R_1$ (chronic CR stim) \n K={}, N={}, $\mu_{{\omega}}$={},  $\sigma_{{\omega}}$={}'.format(K,N,mean_w,std_w))

R4fig, R4ax = plt.subplots()
R4ax.set_xlabel('I')
R4ax.set_ylabel('$\sigma$')
R4ax.set_title('$R_4$ (chronic CR stim) \n K={}, N={}, $\mu_{{\omega}}$={},  $\sigma_{{\omega}}$={}'.format(K,N,mean_w,std_w))

plt.show(block=False)       
R1im = R1ax.imshow(R1s, origin='lower', extent=(Is[0], Is[-1], sigmas[0], sigmas[-1]), aspect='auto')
R1fig.colorbar(R1im)
R4im = R4ax.imshow(R4s, origin='lower', extent=(Is[0], Is[-1], sigmas[0], sigmas[-1]), aspect='auto')
R4fig.colorbar(R4im)



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        