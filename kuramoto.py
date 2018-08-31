import numpy as np
# Note: I'm using globals to keep track of state. Ideally, this should be refactored as object-oriented instead.

def kuramoto(theta, w, N, K):
    """
    Basic Kuramoto model: set of N all-to-all coupled phase oscillators
    theta^dot_j = w_j + K/N sum_{i=1}^N sin(theta_i - theta_j)

    Inputs:
        theta -- phase (Nx1 vector)
        w -- natural frequencies of oscillators (Nx1 vector)
        N -- number of oscillators (scalar)
        K -- coupling strength (scalar)
    Return:
        Nx1 vector -- theta_dot i.e. dtheta/dt
    """   
    return w + K/N*np.sum( np.sin( theta - theta.reshape(N,1) ), axis=1)


def CRstim(t, theta, I, D, rho, P):
    """Coordinated reset stimulation for the Kuramoto model (lysyansky2011)
    S = I * sum_{k=1}^N_c D(x_j, c_k)*rho(t)*P(t)*cos(theta)
    
    Inputs
    ----
        t -- time (scalar)
        theta -- phase (Nx1 vector) 
        I -- stim intensity (scalar)       
        D -- current spatial decay (NxN_c matrix)
        rho -- indicator if stim on at time t (callable rho(t), returns 1xN_c vector)
        P -- burst function (callable P(t), returns scalar)
        N -- number of oscillators (scalar)
    Returns
    ----
        S -- Stim for each oscillator at time t
            (Nx1 vector)
    """
    return I*np.cos(theta)*P(t)*np.sum(D*rho(t), axis=1)
        

def P(t, T_p):
    """Burst function: unit amplitude square wave with period T_cp

    Inputs:
        t -- time (scalar)
        T_p -- 1/[DBS frequency] i.e. intraburst period
    Return:
        Scalar, 1 or 0 -- Value of burst function at time t
    """
    return np.mod(t, T_p) < T_p/2


def rho(t, N_c, T):
    """Indicator function, tells which contact is on at time t. 
    Cycles through contacts in order, switches every T/N_c seconds 

    Inputs:
        t -- time (scalar or vector)
        N_c -- number of contacts (scalar)
        T -- stimulation cycle length, i.e. 1/[CR frequency], i.e. intraburst period (scalar)

    Returns:
        len(t)xN_c vector -- Entry k is indicator (1 or 0) whether contact k is ON or OFF
    """
    return  (np.arange(0, N_c, dtype=float)*T/N_c <= np.mod(t,T).reshape(-1,1))  \
          & (np.mod(t,T).reshape(-1,1) < np.arange(1, N_c+1, dtype=float)*T/N_c)
    
    
def rhoMN(t, N_c, m, n, T, Tmn, Tm):   
    """Indicator function tells which contact is on at time t and if simulation is ON 
    during m:n ON/OFF stim, as in Lysyansky 2011. Starts with contact 1 at beginning of
    every ON period, then cycles in order until OFF.
    
    Note: this is NOT the same as rho(t,N_c,T)*MNflash(t, Tmn, Tm), where the ON/OFF and
    contact cycling are completely independent.

    Inputs:
        t -- time (scalar)
        N_c -- number of contacts (scalar)
        m -- number ON periods (scalar)
        n -- number OFF periods (scalar)
        T -- stimulation cycle length, i.e. 1/[CR frequency], i.e. intraburst period (scalar)
        Tmn -- (m+n)*T
        Tm -- m*T
    Returns:
        1xN_c vector -- Entry k is indicator (1 or 0) whether contact k is ON or OFF
    """
    #Although I can calculate Tmn and Tm inside here, pass them in pre-multiplied to avoid re-doing the multiplication
    return   (np.arange(0, N_c, dtype=float)*T/N_c <= np.mod(np.mod(t,Tmn),T).reshape(-1,1))  \
           & (np.mod(np.mod(t,Tmn),T).reshape(-1,1) < np.arange(1, N_c+1, dtype=float)*T/N_c) \
           & (0 <= np.mod(t,Tmn).reshape(-1,1)) \
           & (np.mod(t,Tmn).reshape(-1,1) < Tm)

 
def rhoPerm(t, N_c, T, T_switch):
    """Indicator function, tells which contact is on at time t. 
    Randomly permuted contact ordering, a new contact is selected every T/N_c seconds. Guaranteed that 
    each of the N_c contacts is used exactly once over a period of T seconds. After T_switch seconds, a 
    new permutation is randomly generated. 
       
    Note: rhoPerm(t, N_c, T, T_switch=T/N_c) is equivalent to rhoRand(t, N_c, T, repeats=True), 
    although rhoRand should probably be faster in this case     
             
    Inputs:
        see rho(t, N_c, T), except t must be scalar
        T_switch -- duration for which a permutation is used before a new one is selected. Recommended to be multiple of T (scalar)
    Returns:
        see rho(t, N_c, T)
    """
    global RHO_PERM_STATE #keeps track of the current permutation
    global RHO_PERM_LAST_UPDATE #keeps track of the last time state was updated
    if not 'RHO_PERM_LAST_UPDATE' in globals():
        RHO_PERM_LAST_UPDATE = t
        RHO_PERM_STATE = np.random.permutation(np.arange(N_c, dtype=float))  
    if ((t-RHO_PERM_LAST_UPDATE)>=T_switch and np.isclose(np.mod(t,T_switch), 0)):
        RHO_PERM_LAST_UPDATE = t
        RHO_PERM_STATE = np.random.permutation(np.arange(N_c, dtype=float))  
    return  (RHO_PERM_STATE*T/N_c <= np.mod(t,T).reshape(-1,1)) & (np.mod(t,T).reshape(-1,1) < (RHO_PERM_STATE+1)*T/N_c)


def MNflash(t, Tmn, Tm):
    """Indicator function whether stimulation is ON of OFF at time t
    Stimulation is ON for m periods and OFF for n periods of length T
    """
    return   (0 <= np.mod(t,Tmn)) & (np.mod(t,Tmn) < Tm)
   
 
def order_param(theta, N, k=1):
    """Returns kth order parameter
    """
    Rk = np.sum( np.exp(1j * k * theta.reshape(-1,N)), axis=1 )/N
    return np.abs(Rk), np.angle(Rk)


def mean_max_R(R, N_rp, T, m, n, first_rp=None, t=None):
    """Return the mean-max of the order parameter R, defined as <r> = 1/N_rp sum_{i=1}^N_rp r_i
    where r_i is the max value of R during the OFF period and N_rp is the number of rest periods
    """
    Tmn = (m+n)*T
    Tm = m*T    
    if first_rp is None:
        first_rp = np.round(N_rp/2) #assume reached steady-state after [N_rp/2] ON-OFF periods
    if t is None:
        t0 = 0
        tf = N_rp*Tmn 
        tstep = 0.001
        t = np.arange(t0, tf, tstep)
    
    mean_max_R = 0 #mean max of R (mean over all rest periods, max within rest period)
    for rp in range(first_rp, N_rp): #iterate through rest periods
        idx = (rp*Tmn+Tm <=t)&(t< (rp+1)*Tmn)
        ri = np.max( R[idx] )
        mean_max_R += ri
    mean_max_R = mean_max_R/(N_rp - first_rp)
    return mean_max_R


def rk4(t, h, y, f):
    """Takes one step using fourth-order Runge-Kutta numerical integration
    
    Inputs:
        t -- integration time start (scalar)
        h -- integration step size (scalar)
        y -- value of y at time t (scalar or Nx1 where N is number of ODEs in system)
        f -- rate of change of y at time t, i.e. dy/dt = f(t,y) (function handle)
    Returns:
        scalar or Nx1 vector -- value of y at time (t+h)
    """
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5*h, y + 0.5*k1)
    k3 = h * f(t + 0.5*h, y + 0.5*k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*(k2 + k3) + k4)/6.0

