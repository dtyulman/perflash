from kuramoto import kuramoto, order_param
from scipy.integrate import ode
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import timeit

"""Run the Kuramoto model without stimulation with varying coupling constants, and plot order parameter.
"""

# Model parameters
N = 200                                #number of oscillators in system
K = None #will loop over this          #coupling strength
mean_w = pi                            #mean of distribution of natural frequencies of oscillators
std_w = 0.02                           #standard deviation of disribution of natural freqs
w = np.random.normal(mean_w, std_w, N) #natural frequencies normally distributed

# Integration parameters
theta0 = np.linspace(0, 2*pi, N, endpoint=False) #initial phases of oscillators
t0 = 0.                                          #integration start time
tf = 100.                                        #integration end time
tstep = 0.001                                    #time step

for K in [0.0001, 0.005, 0.01, 0.03, 0.04, 0.1, 0.5, 1.]:
    # Set up the solver
    kursolver = ode(lambda t, theta: kuramoto(theta,w,N,K))
    kursolver.set_integrator('vode', method='bdf')
    kursolver.set_initial_value(theta0, t0)
    
    # Allocate memory and run the solver
    t = np.arange(t0, tf, tstep)
    theta = np.full((t.size, N), np.nan) #each row is time step  
    r = np.full(t.shape, np.nan) 
    
    theta[0,:] = theta0
    r[0] = order_param(theta[0,:], N)[0] 
    
    print('Starting K={}'.format(K))
    timeitstart = timeit.default_timer()
    for i in range(1,t.size):
        theta[i,:] = kursolver.integrate(t[i]) #use SciPy solver
        if not kursolver.successful():
           raise Exception('Integration step unsuccessful')
        r[i], _ = order_param(theta[i,:], N)
    print('Elapsed time: {} sec'.format(timeit.default_timer()-timeitstart))
    
    timeitstart = timeit.default_timer()
    fig = plt.figure()
    plt.plot(t,r)
    plt.title('First order parameter \n K={0}, N={1}, $\mu_{{\omega}}$={2},  $\sigma_{{\omega}}={3}$'.format(K,N,mean_w,std_w))
    plt.xlabel('Time (s)')
    plt.ylabel('$R_1$')
    plt.show(block=False)
    plt.pause(0.001)
    print('Time to plot: {} sec'.format(timeit.default_timer()-timeitstart))












