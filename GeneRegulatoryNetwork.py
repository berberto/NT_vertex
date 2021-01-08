#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:26:22 2020

@author: andrewg
"""
import numpy as np
from scipy.integrate import odeint
import time

def bistable(x,y,M,alpha,sig_x,sig_y,K_x,K_y, K_M , f):
    return [(1+sig_x*((1+M/K_M)/(1.0+ f*M/K_M ))**2 )**-1 - x, alpha*(1+ sig_y*(1+ x/K_x)**2)**-1] 

def bistable_GRN_fn(z,t): #if.cond.. x,y,w=z,  return lorentz(x,y,w)
    x,y=z
    return bistable(x,y,1,1,1,1,1,1,1,1) 

def bistable_GRN_input(z,t,input_signal,params):
    #p is a vector of parameters
    alpha,sig_x,sig_y,K_x,K_y, K_M , f = params #set parameters, dim(p)=7
    x,y=z
    return bistable(x,y,input_signal,alpha,sig_x,sig_y,K_x,K_y, K_M ,f)

def bistable_input(z,t,input_signal):
    alpha,sig_x,sig_y,K_x,K_y, K_M , f = np.ones(7) #to be modified for standard parameters
    x,y=z
    return bistable(x,y,input_signal,alpha,sig_x,sig_y,K_x,K_y, K_M ,f)

def balaskas(p,o,n,G,par): #p is the vector of parameters
    alpha,beta,gamma,NcO,NcP,OcP,OcN,PcN,k1,k2,k3,h1,h2,h3,h4,h5=par
    return [(alpha / (1.0 + (n/NcP)**h1 + (o/OcP)**h2) ) - k1*p,(beta*G/(1+G))*(1.0/(1+(n/NcO)**h3 )) - k2*o,(gamma*G/(1+G))*(1.0/(1+(o/OcN)**h4 + (p/PcN)**h5)) - k3*n ]

def balaskas_input(z,t,input_signal):
    suitable_paras=np.ones(16) #to be set correctly.
    p,o,n = z
    return balaskas(p,o,n,input_signal,suitable_paras)

def poni(p,o,n,i,a,r,par1,parP,parG,parPax,parO,parN,parI):
    alpha, beta, c_AP,P=par1
    K_P_Pax,K_P_O,K_P_N,K_P_I =parP
    K_G_O,K_G_N = parG
    K_Pax_N=parPax[0] #because it's 1d
    K_O_Pax,K_O_N,K_O_I = parO
    K_N_Pax,K_N_O,K_N_I = parN
    K_I_O,K_I_N = parI
    Z_pax = K_P_Pax*P + ((1.0+K_O_Pax*o)**2)*(1+K_N_Pax*n)**2 #ok
    Z_I=K_P_I*P + ((1.0+K_O_I*o)**2)*(1+K_N_I*n)**2 #ok
    Z_O=K_P_O*P + c_AP*K_P_O*P*K_G_O*a + (1+K_G_O*r + K_G_O*a)*((1+K_N_O*n)**2)*((1+K_I_O*i)**2) #ok
    Z_N = K_P_N*P + c_AP*K_P_N*P*K_G_N*a + (1+K_G_N*r + K_G_N*a)*((1+K_O_N*o)**2)*((1+K_I_N*i)**2)*(1+K_Pax_N*p)**2 #ok
    phiPax = K_P_Pax*P/Z_pax #ok
    phiI=K_P_I*P/Z_I #ok
    phiO=(K_P_O*P + c_AP*K_P_O*P*K_G_O*a) / Z_O #ok
    phiN =(K_P_N*P + c_AP*K_P_N*P*K_G_N*a) / Z_N #ok
    return np.array([alpha*phiPax  - beta*p,alpha*phiO - beta*o , alpha*phiN - beta*n , alpha*phiI - beta*i]) #ok
    
def poni_input(z,t,input_signal,par):
    p,o,n,i = z
    par1=par[0] #dim is 4
    parP = par[1] #dim is 4
    parG = par[2] #dim is 2
    parPax = par[3] # dim 1
    parO=par[4] # dim 3
    parN = par[5] #dim 3
    parI = par[6] #dim 2
    a = input_signal #or some function of SHH 
    r = 1 - a
    return poni(p,o,n,i,a, r , par1,parP,parG,parPax,parO,parN,parI)

def poni_input_default(z,t,input_signal):
    """
    alpha, beta taken to be 0.34 h^-1, corresponding to a half life of 2 hours.
    In machine time units, this corresponds to 0.045 (tu)^-1, approx.
    """
    p,o,n,i= z
    par1=[2,2,10,0.8] #dim is 4,alpha, beta, c_AP,P. alpha,beta were 0.045 
    parP = [4.8,47.8,27.4,23.4] #dim is 4,K_P_Pax,K_P_O,K_P_N,K_P_I 
    parG = [18.0,37.30] #old values[18.0,37.3] #dim is 2,K_G_O,K_G_N
    parPax = [4.8] # dim 1,  K_Pax_N
    parO=[1.9,27.1,58.8] # old values [1.9,27.1,58.8] dim 3, K_O_Pax,K_O_N,K_O_I
    parN = [26.7,60.6,76.2] # old values [26.7,60.6,76.2] dim 3 K_NPax, K_NO,K_NI
    parI = [28.4,47.1] #dim 2, K_IO, K_IN
    a = input_signal #or some function of SHH 
    r = 1 - a
    return poni(p,o,n,i,a, r , par1,parP,parG,parPax,parO,parN,parI)

def ptch_to_gli(ptc,PtcIn,Ptc,x,X,gliFL,GliFL,GliA,GliR,gfp,deg,k ,K,Km,POL,tl,tr,act,conv,c,Shh):
    d_GliFL,d_GliR, d_GliA,d_Ptc,d_X,d_gfp,d_gliFL,d_ptc,d_x = deg #degradation rates
    k_ShhPtc = k[0] #k is a list of one elt
    K_Gli_gfp, K_Gli_ptc, K_Gli_x, K_X_gli, K_Pol_gfp, K_Pol_gli, K_Pol_ptc, K_Pol_x = K
    Km_Ptc = Km[0]
    Pol=POL[0]
    tl_GliFL,tl_Ptc,tl_X= tl
    tr_gfp,tr_gliFL,tr_ptc,tr_x = tr
    act_Ptc=act[0]
    conv_GliA, conv_GliR = conv
    c_GliA , c_X = c
    d_ptc_denom_pt1 = K_Pol_ptc*Pol*(K_Gli_ptc*GliA+1)**2 +(K_Gli_ptc*GliA)**2 +(K_Gli_ptc*GliR)**2
    d_ptc_denom_pt2=+2*K_Gli_ptc*GliR +2*K_Gli_ptc*GliA +2* K_Gli_ptc*GliR*K_Gli_ptc*GliA +1
    d_x_denom_pt1 = K_Pol_x*Pol*(K_Gli_x*GliA+1)**2 +(K_Gli_x*GliA)**2 +(K_Gli_x*GliR)**2 
    d_x_denom_pt2=+2*K_Gli_x*GliR +2*K_Gli_x*GliA +2* K_Gli_x*GliR*K_Gli_x*GliA +1
    d_gfp_denom_pt1 = K_Pol_gfp*Pol*(K_Gli_gfp*GliA+1)**2 +(K_Gli_gfp*GliA)**2 +(K_Gli_gfp*GliR)**2 
    d_gfp_denom_pt2=+2*K_Gli_gfp*GliR +2*K_Gli_gfp*GliA +2* K_Gli_gfp*GliR*K_Gli_gfp*GliA +1
    deriv_ptc = (tr_ptc*((K_Gli_ptc*GliA*c_GliA + 1)**2)*K_Pol_ptc*Pol)/(d_ptc_denom_pt1 + d_ptc_denom_pt2)  - d_ptc*ptc
    deriv_x = (tr_x*((K_Gli_x*GliA*c_GliA + 1)**2)*K_Pol_x*Pol)/(d_x_denom_pt1 + d_x_denom_pt2)  - d_x*x
    deriv_gfp = (tr_gfp*((K_Gli_gfp*GliA*c_GliA + 1)**2)*K_Pol_gfp*Pol)/(d_gfp_denom_pt1 + d_gfp_denom_pt2)  - d_gfp*gfp
    deriv_PtcIn = tl_Ptc*ptc - d_Ptc*PtcIn - act_Ptc*PtcIn
    deriv_Ptc =  act_Ptc*PtcIn - d_Ptc*Ptc -  k_ShhPtc*Ptc*Shh
    deriv_X = tl_X*x - d_X*X
    deriv_gliFL =( tr_gliFL*K_Pol_gli*Pol*(1+K_X_gli*X*c_X)**2) / (1+K_X_gli*X + K_Pol_gli*Pol*(1+K_X_gli*x*c_X)**2) - d_gliFL*gliFL
    deriv_GliFL= tl_GliFL*gliFL - ((conv_GliA*GliFL*Km_Ptc)/(Km_Ptc + Ptc) + conv_GliR*GliFL) - d_GliFL*GliFL
    deriv_GliA = ((conv_GliA*GliFL*Km_Ptc)/(Km_Ptc+Ptc)) - d_GliA*GliA
    deriv_GliR = conv_GliR*GliFL - d_GliR*GliR
    return np.array([deriv_ptc,deriv_PtcIn,deriv_Ptc,deriv_x,deriv_X,deriv_gliFL,deriv_GliFL,deriv_GliA,deriv_GliR,deriv_gfp])

def ptch_gli_input(z,t,Shh,par):
    ptc,PtcIn,Ptc,x,X,gliFL,GliFL,GliA,GliR,gfp = z
    deg = par[0] #len is 9
    k=par[1] #len 1
    K=par[2]#len 8
    Km==par[3] #len 1
    POL=par[4]#len 1
    tl=par[5] #len 3
    tr=par[6]#len 4
    act=par[7]# len 1
    conv=par[8]#len 2
    c=par[9]#len 2
    return ptch_to_gli(ptc,PtcIn,Ptc,x,X,gliFL,GliFL,GliA,GliR,gfp,deg,k ,K,Km,POL,tl,tr,act,conv,c,Shh)

def ptch_gli_default(z,t,Shh):
    ptc,PtcIn,Ptc,x,X,gliFL,GliFL,GliA,GliR,gfp = z
    deg = [0.1,0.01,1.5,0.1,0.5,1.4,0.03,2.0,1.0] # d_GliFL,d_GliR, d_GliA,d_Ptc,d_X,d_gfp,d_gliFL,d_ptc,d_x
    k=[100.0] # k_ShhPtc
    K=[2.5,6.3,3.98,10.0,0.63,0.01,1.0,1.0]# K_Gli_gfp, K_Gli_ptc, K_Gli_x, K_X_gli, K_Pol_gfp, K_Pol_gli, K_Pol_ptc, K_Pol_x
    Km=[1.0] # Km_Ptc
    POL=[1.0]# Pol
    tl=[100.0,100.0,1.0] # tl_GliFL,tl_Ptc,tl_X
    tr=[1.22, 12.59,28.18,0.79]# tr_gfp,tr_gliFL,tr_ptc,tr_x
    act=[10.0]# act_Ptc
    conv=[1.22, 0.71]# conv_GliA, conv_GliR
    c=[10.0,0.0]# c_GliA , c_X.   c_X = 0 for adaptation? Check Cohen, Kicheva et al.
    return ptch_to_gli(ptc,PtcIn,Ptc,x,X,gliFL,GliFL,GliA,GliR,gfp,deg,k ,K,Km,POL,tl,tr,act,conv,c,Shh)
    
def ptch_gli_default2(z,t,Shh):
    """
    The factor 0.13 multiplying the rate parameters is to convert them for 
    (per hour) to (per machine time)
    
    """
    ptc,PtcIn,Ptc,x,X,gliFL,GliFL,GliA,GliR,gfp = z #d_Glir was 5
    deg = 0.13*np.array([0.1,0.06,0.22,0.1,0.5,1.4,0.03,2.0,1.0]) # d_GliFL,d_GliR, d_GliA,d_Ptc,d_X,d_gfp,d_gliFL,d_ptc,d_x
    k=[100.0] # k_ShhPtc
    K=[2.5,6.3,3.98,10.0,0.63,0.01,1.0,1.0]# K_Gli_gfp, K_Gli_ptc, K_Gli_x, K_X_gli, K_Pol_gfp, K_Pol_gli, K_Pol_ptc, K_Pol_x
    Km=[1.0] # Km_Ptc
    POL=[1.0]# Pol
    tl=0.13*np.array([100.0,100.0,1.0]) # tl_GliFL,tl_Ptc,tl_X
    tr=0.13*np.array([1.22, 12.59,28.18,0.79])# tr_gfp,tr_gliFL,tr_ptc,tr_x
    act=[10.0]# act_Ptc
    conv=0.13*np.array([1.22, 0.71])# conv_GliA, conv_GliR
    c=[10.0,1.0]# c_GliA , c_X.   c_X = 0 for adaptation? Check Cohen, Kicheva et al.
    return ptch_to_gli(ptc,PtcIn,Ptc,x,X,gliFL,GliFL,GliA,GliR,gfp,deg,k ,K,Km,POL,tl,tr,act,conv,c,Shh)   
    
def ptch_gli_default2_sc(z,t,Shh, scale):
    """
    scale converts the rates (per hour) to rates (per machine time unit).
    scale will almost always be 0.13 as one machine time unit corresponds to
    0.13 hours.
    """
    ptc,PtcIn,Ptc,x,X,gliFL,GliFL,GliA,GliR,gfp = z #d_Glir was 5
    deg = scale*np.array([0.1,0.06,0.22,0.1,0.5,1.4,0.03,2.0,1.0]) # d_GliFL,d_GliR, d_GliA,d_Ptc,d_X,d_gfp,d_gliFL,d_ptc,d_x
    k=[100.0] # k_ShhPtc
    K=[2.5,6.3,3.98,10.0,0.63,0.01,1.0,1.0]# K_Gli_gfp, K_Gli_ptc, K_Gli_x, K_X_gli, K_Pol_gfp, K_Pol_gli, K_Pol_ptc, K_Pol_x
    Km=[1.0] # Km_Ptc
    POL=[1.0]# Pol
    tl=scale*np.array([100.0,100.0,1.0]) # tl_GliFL,tl_Ptc,tl_X
    tr=scale*np.array([1.22, 12.59,28.18,0.79])# tr_gfp,tr_gliFL,tr_ptc,tr_x
    act=[10.0]# act_Ptc
    conv=scale*np.array([1.22, 0.71])# conv_GliA, conv_GliR
    c=[10.0,1.0]# c_GliA , c_X.   c_X = 0 for adaptation? Check Cohen, Kicheva et al.
    return ptch_to_gli(ptc,PtcIn,Ptc,x,X,gliFL,GliFL,GliA,GliR,gfp,deg,k ,K,Km,POL,tl,tr,act,conv,c,Shh) 




class GRN3(object):
    """
    Attributes:
        state[i] is the state vector of the dynamical system in face i
        GRN function is the right hand side of the dynamical system in each face
        GRN function will depend on z (current state), t time and an input (float)
        parameter due to the morphogen signal.
        sig_input is an array.  sig_input[i] are the input parameters due to the 
        signal for the GRN in cell i.  At the moment, sig_input[i] is one dimensional.
        inherited is a vector of the same length as the state vector. 
        inhetited[i] belongs to the interval (0,1).  It is the proportion of 
        the ith protein which it inherits from its parent.
    """
    
    def __init__(self,state,GRN_function,sig_input,inherited):
        self.state=state #state[i] is the state of the GRN
        self.GRN_function=GRN_function #GRN function for all cells
        self.sig_input=sig_input 
        self.inherited = inherited
        
    def evolve_GRN(self,time,dt,sig_input, living_faces): #could put self.sig_input...
        """
        dt is the time step
        morph_input[i] is the parameter input to cell[i]
        living_faces is an array of indices of faces which are currently alive
        
        """
        ic= self.state  
        t= np.linspace(time,time+dt,2) #linspace of two elts.  
        for k in living_faces:
            self.state[k] = odeint(self.GRN_function , ic[k], t , args=(sig_input[k],))[-1] #new state vector for each face
    
    def evolve_GRN2(self,time,dt,living_faces): #could put self.sig_input...
        """
        dt is the time step
        morph_input[i] is the parameter input to cell[i]
        living_faces is an array of indices of faces which are currently alive
        
        """
        ic= self.state  
        sig_input = self.sig_input
        t= np.linspace(time,time+dt,2) #linspace of two elts.  
        for k in living_faces:
            self.state[k] = odeint(self.GRN_function , ic[k], t , args=(sig_input[k],))[-1] #new state vector for each face
    
    def evolve_GRN_fine(self,time,k,dt,sig_input, living_faces):
        """
        dt is the time step
        sig_input[i] is the parameter input to cell[i]
        living_faces is an array of indices of faces which are currently alive
        k >=0 is the number of extra steps over the time interval dt
        
        """
        ic= self.state  
        t= np.linspace(time,time+dt,k+1) #linspace of two elts.  
        for j in living_faces:
            self.state[j] = odeint(self.GRN_function , ic[j], t , args=(sig_input[j],))[-1] #new state vector for each face   
        
        
def build_GRN3(n_face, sys=None, state=None , sig_input=None, inherited=None): #used in NT_full_sim_seq
    """
    n_face is the number of faces in the grid
    if sys is None, we insert the PONI GRN, with default parameters
       if sys is 'b', we use the balaskas GRN and otherwise it's the bistable_switch.
       if sys is 'p', we use the Cohen transduction GRN
    if state is None, we set each GRN state to zero to start.  Otherwise, care
    must be taken that the dimensions of state are appropriate to the chosen
    GRN_function.
    """
    if sys is None:
        GRN_function = poni_input_default
        if state is None:
            state= np.array([[0.317,0.0013,0.000007,0.94],]*n_face) #approx steady state
        else: 
            state=state
        if sig_input is None: 
            sig_input = np.zeros(n_face) #no signal to start
        else:
            sig_input = sig_input
        if inherited is None:
            
            inherited = np.ones((n_face,4))
        else:
            inherited = inherited
    elif sys is "p":
        GRN_function = ptch_gli_default2
        if state is None:
            state= np.array([[0.000015,0.00015,1623.0,0.00016,0.051,1.0,118.11,0.054,687.57,0.0000040],]*n_face)#np.array([[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],]*n_face) #change to a steady state start value
        else: 
            state=state
        if sig_input is None: 
            sig_input = np.zeros(n_face) #no signal to start
        else:
            sig_input = sig_input
        if inherited is None:
            inherited = np.ones((n_face,10))
        else:
            inherited = inherited
    elif sys is "b":
        GRN_function = balaskas_input
        if state is None:
            state= np.ones((n_face, 3))
        else: 
            state=state
        if sig_input is None:
            sig_input = np.zeros(n_face)
        else:
            sig_input = sig_input
        if inherited is None:
            inherited = np.ones((n_face,3))
        else:
            inherited = inherited
    else:
        GRN_function = bistable_input
        if state is None:
            state= np.ones((n_face, 2))
        else: 
            state=state
        if sig_input is None:
            sig_input = np.zeros(n_face)
        else:
            sig_input = sig_input   
        if inherited is None:
            inherited = np.ones((n_face,2))
        else:
            inherited = inherited
    return GRN3(state, GRN_function, sig_input, inherited) 


class GRN_basic(object):
    """
    Attributes:
        state[i] is the state vector of the dynamical system in face/node i.

        GRN function is the right hand side of the dynamical system in each face
        GRN function will depend on z (current state), t time and an input (float)
        parameter.
        sig_input is an array.  sig_input[i] are the input parameters due to the 
        signal for the GRN in cell i.  
        
        SHOULD ADD 'inherited' attribute for a JDS history if we want to include
        uneven splits.
    """
    
    def __init__(self,state,GRN_function,sig_input):
        self.state=state #state[i] is the state of the GRN
        self.GRN_function=GRN_function #GRN function for all cells
        self.sig_input=sig_input 
        
    def evolve_GRN(self,time,dt,sig_input): #could put self.sig_input...
        """
        dt is the time step
        morph_input[i] is the parameter input to cell[i]
        living_faces is an array of indices of faces which are currently alive
        
        
        
        """
        ic= self.state  
        t= np.linspace(time,time+dt,2) #linspace of two elts.  
        for k in range(len(self.state)):
            self.state[k] = odeint(self.GRN_function , ic[k], t , args=(sig_input[k],))[-1] #new state vector for each face
    
    def evolve_GRN_subset(self,time,dt,sig_input,ids_to_evolve):
        """
        dt is the time step
        morph_input[i] is the parameter input to cell[i]
        living_faces is an array of indices of faces which are currently alive
        
        """
        ic= self.state  
        t= np.linspace(time,time+dt,2) #linspace of two elts.  
        for k in range(len(self.state)):
            if k in ids_to_evolve:
                self.state[k] = odeint(self.GRN_function , ic[k], t , args=(sig_input[k],))[-1] #new state vector for each face
            
    
    def evolve_GRN2(self,time,dt): #could put self.sig_input...
        """
        dt is the time step
        morph_input[i] is the parameter input to cell[i]
        
        
        """
        ic= self.state  
        sig_input = self.sig_input
        t= np.linspace(time,time+dt,2) #linspace of two elts.  
        for k in range(len(self.state)):
            self.state[k] = odeint(self.GRN_function , ic[k], t , args=(sig_input[k],))[-1] #new state vector for each face
    
    def evolve_GRN_fine(self,time,k,dt,sig_input):
        """
        dt is the time step
        sig_input[i] is the parameter input to cell[i]
        living_faces is an array of indices of faces which are currently alive
        k >=0 is the number of extra steps over the time interval dt
        
        """
        ic= self.state  
        t= np.linspace(time,time+dt,k+1) #linspace of two elts.  
        for j in range(len(self.state)):
            self.state[j] = odeint(self.GRN_function , ic[j], t , args=(sig_input[j],))[-1] #new state vector for each face   
   
    def evolve_ugly(self,time,dt,sig_input):
        """
        dt is the time step
        sig_input[i] is the parameter input to cell[i]
        living_faces is an array of indices of faces which are currently alive
        k >=0 is the number of extra steps over the time interval dt
        
        """
        ic= self.state  
        for j in range(len(self.state)):
            self.state[j] = ic[j] + dt*np.array(self.GRN_function(ic[j],time,sig_input[j])) #new state vector for each face 
            
    def evolve_ugly_2(self,time,dt,sig_input):
        """
        dt is the time step
        sig_input[i] is the parameter input to cell[i]
        living_faces is an array of indices of faces which are currently alive
        k >=0 is the number of extra steps over the time interval dt
        
        """
        ic= self.state  
        self.state = ic + dt*self.GRN_function(ic,time,sig_input) #new state vector for each face 
    
    
    
    def divide(self,ready):
        for k in ready:#each new cell gets half of the parent's cell state.
            daughter_state= 0.5*self.state[k] 
            old_state = self.state
            new_state = np.vstack((old_state,daughter_state))
            self.state = np.vstack((new_state, daughter_state))
            
        
        
def build_GRN_basic(n, sys=None, state=None , sig_input=None): #used in NT_full_sim_seq
    """
    n is the number of GRNs to be simulated 
    if sys is None, we insert the PONI GRN, with default parameters
       if sys is 'b', we use the balaskas GRN and otherwise it's the bistable_switch.
    if state is None, we set each GRN state to zero to start.  Otherwise, care
    must be taken that the dimensions of state are appropriate to the chosen
    GRN_function.
    """
    if sys is None:
        GRN_function = poni_input_default
        if state is None:
            state= np.array([[0.317,0.0013,0.000007,0.94],]*n) #approx steady state
        else: 
            state=state
        if sig_input is None: 
            sig_input = np.zeros(n) #no signal to start
        else:
            sig_input = sig_input
    elif sys is "p":
        GRN_function = ptch_gli_default2
        if state is None:
            state= np.array([[0.000015,0.00015,1623.0,0.00016,0.051,1.0,118.11,0.054,687.57,0.0000040],]*n)#np.array([[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],]*n) #change to a steady state start value
        else: 
            state=state
        if sig_input is None: 
            sig_input = np.zeros(n) #no signal to start
        else:
            sig_input = sig_input
    elif sys is "b":
        GRN_function = balaskas_input
        if state is None:
            state= np.ones((n, 3))
        else: 
            state=state
        if sig_input is None:
            sig_input = np.zeros(n)
        else:
            sig_input = sig_input
    else:
        GRN_function = bistable_input
        if state is None:
            state= np.ones((n, 2))
        else: 
            state=state
        if sig_input is None:
            sig_input = np.zeros(n)
        else:
            sig_input = sig_input   
    return GRN_basic(state, GRN_function, sig_input) 


def compute_effector(trans_grn):
    """
    Computes and returns the effector, which is the input to the poni_grn.
    
    """
    glia=trans_grn.state[:,7]
    glir=trans_grn.state[:,8]
    gli=glia+glir
    n=len(trans_grn.state)
    effector = np.zeros(n) # effector
    for k in range(n):
        if gli[k] > 0.0000001:
            effector[k] = glia[k]/gli[k] #taking effector value to be proportion of glia
    return effector 

class GRN_full_basic(object):
    def __init__(self, poni_grn, trans_grn, effector, lost_morphogen):
        self.poni_grn = poni_grn # a GRN_basic object
        self.trans_grn = trans_grn #a GRN_basic object
        self.effector = effector
        self.lost_morphogen = lost_morphogen
           
        
    def evolve(self, time , dt , sig_input , bind_rate): #bind_rate = k_ShhPtc from Cohen.  Make sure the scale is correct.
        """
        Args:
            time is the current time
            dt is the time step
            sig_input is a vector s.t. sig_input[i] is the input grn[i]
        """
        self.lost_morphogen = bind_rate * sig_input*self.trans_grn.state[:,2] #to be appropriately modified and subtracted from the concentration vector.
        self.trans_grn.evolve_GRN(time,dt,sig_input)
        self.poni_grn.evolve_GRN(time,dt,self.effector)
        self.effector = compute_effector(self.trans_grn) #update time?
        
    def evolve_ugly(self, time , dt , sig_input , bind_rate): #bind_rate = k_ShhPtc from Cohen.  Make sure the scale is correct.
        """
        Args:
            time is the current time
            dt is the time step
            sig_input is a vector s.t. sig_input[i] is the input grn[i]
        """
        self.lost_morphogen = bind_rate * sig_input*self.trans_grn.state[:,2] #to be appropriately modified and subtracted from the concentration vector.
        self.trans_grn.evolve_ugly(time,dt,sig_input)
        self.poni_grn.evolve_ugly(time,dt,self.effector)
        self.effector = compute_effector(self.trans_grn) #update time?
        
    def evolve_subset(self, time , dt , sig_input , bind_rate, ids_to_evolve):
        #bind_rate = k_ShhPtc from Cohen.  Make sure the scale is correct.
        self.lost_morphogen = bind_rate * sig_input*self.trans_grn.state[:,2] #to be appropriately modified and subtracted from the concentration vector.
        self.trans_grn.evolve_GRN_subset(time,dt,sig_input,ids_to_evolve)
        self.poni_grn.evolve_GRN_subset(time,dt,self.effector,ids_to_evolve)
        self.effector = compute_effector(self.trans_grn) #update time?
        
    
    def division(self,ready):
        self.trans_grn.divide(ready)
        self.poni_grn.divide(ready)
        self.effector = compute_effector(self.trans_grn)
        #lost morphogen is reset after an evolution step and subtracted.  No need to update.
        
        
        
            
         
def build_GRN_full_basic(n):
    poni = build_GRN_basic(n)
    trans = build_GRN_basic(n,"p")
    effector = compute_effector(trans)
    lost_morph = np.zeros(n)
    return GRN_full_basic(poni, trans, effector, lost_morph)


if __name__ == "__main__":
    test_g = build_GRN_full_basic(10)
    t1=time.time()
    test_g.evolve_ugly(1.0,0.001,np.random.normal(0,1,10), 0.01)
    t2=time.time()

    t3=time.time()
    test_g.evolve(1.0,0.001,np.random.normal(0,1,10), 0.01)
    t4=time.time()

    print("odeint took ", t4 - t3)
    print("basic took ", t2 - t1)
    print("basic is ", (t4 - t3)/(t2 - t1), " times quicker.")

        


     
      