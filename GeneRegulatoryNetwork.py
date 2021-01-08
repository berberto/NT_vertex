#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:26:22 2020

@author: andrewg
"""
import numpy as np
from scipy.integrate import odeint
import time



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

        


     
      