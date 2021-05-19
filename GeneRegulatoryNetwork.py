#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:26:22 2020

@author: andrewg
"""
import numpy as np
from scipy.integrate import odeint
from constants import time_hours, diff_rate_hours
import time

idx={
    "Ptc":  0,
    "GliF": 1,
    "GliA": 2,
    "GliR": 3,
    "Pax":  4,
    "Oli":  5,
    "Nkx":  6,
    "Irx":  7
}
nSpecies = len(idx)

def Hill (x):
    return x / ( 1. + x )

f_A = 10.
c_GliA = 10.


# using parameter set 152 in shh_LAX_long
def shh_gli_poni (X, t, shh,
      # Shh-Gli
      K_Pol_Ptc=0.206020,
      K_Pol_Gli=30.413137,
      K_Gli_Ptc=0.354320,
      Km_Ptc_Gli=293.792540,
      alpha_Ptc=0.041105,
      alpha_GliF=0.522472,
      w_Gli_A=1.811713,
      w_Gli_R=0.224316,
      delta_Ptc=0.003679,
      delta_GliF=0.143287,
      delta_GliA=0.771230,
      delta_GliR=0.223370,
      B_Shh_Ptc=161.192033,
      # PONI
      alpha_Pax=1.636826,
      alpha_Oli=2.045280,
      alpha_Nkx=2.365557,
      alpha_Irx=2.192375,
      delta=2.017940,
      K_Pol_Pax=4.397879,
      K_Oli_Pax=1.909479,
      K_Nkx_Pax=26.534532,
      K_Pol_Oli=56.228966,
      K_Gli_Oli=16.200857,
      K_Nkx_Oli=64.417153,
      K_Irx_Oli=29.268699,
      K_Pol_Nkx=25.486457,
      K_Gli_Nkx=345.067599,
      K_Pax_Nkx=4.883786,
      K_Oli_Nkx=30.411750,
      K_Irx_Nkx=56.370927,
      K_Pol_Irx=24.712558,
      K_Oli_Irx=59.936165,
      K_Nkx_Irx=78.009027,
      # Feedback
      K_Nkx_Gli=8.241428,
      f_Nkx_Gli=0.102894,
      K_Oli_Gli=7.422326,
      f_Oli_Gli=0.141334
     ):
    '''
    Dynamical system of the full model
    '''
    
    ptc,   glif,   glia,   glir,   pax,   oli,   nkx,   irx = X
    f_ptc, f_glif, f_glia, f_glir, f_pax, f_oli, f_nkx, f_irx = np.zeros(X.shape)
    
    # feedback weights
    aux_1 = 1 + f_Oli_Gli * K_Oli_Gli * oli
    aux_1 *= 1 + f_Nkx_Gli * K_Nkx_Gli * nkx
    aux_2 = 1 + K_Oli_Gli * oli
    aux_2 *= 1 + K_Nkx_Gli * nkx

    feedbackPtc = 1.
    feedbackGli = (aux_1/aux_2)*(aux_1/aux_2)
    
    # Ptc
    aux_1 = 1. + c_GliA * K_Gli_Ptc * glia
    aux_2 = 1. + K_Gli_Ptc*(glia + glir)
    aux_3 = aux_1*aux_1 / ( aux_2*aux_2 ) * feedbackPtc
    f_ptc = alpha_Ptc * Hill( K_Pol_Ptc * aux_3 )
    f_ptc += - ( delta_Ptc + B_Shh_Ptc * shh ) * ptc

    # GliFL
    aux_1 = feedbackGli
    f_glif = alpha_GliF * Hill( K_Pol_Gli * aux_1 )
    f_glif += - ( w_Gli_A / ( 1. + Km_Ptc_Gli * ptc ) + w_Gli_R + delta_GliF ) * glif

    # GliA
    f_glia = w_Gli_A / ( 1. + Km_Ptc_Gli * ptc ) * glif
    f_glia += - delta_GliA * glia

    # GliR
    f_glir = w_Gli_R * glif
    f_glir += - delta_GliR * glir

    # Pax
    # Repression by Olig
    aux_1 = 1./(1. + K_Oli_Pax * oli)
    aux_1 *= aux_1
    # Repression by Nkx
    aux_2 = 1./(1. + K_Nkx_Pax * nkx)
    aux_2 *= aux_2
    f_pax = alpha_Pax * Hill(K_Pol_Pax  * aux_1 * aux_2)
    f_pax += - delta * pax

    # Olig
    # Activation by Gli
    aux_1 = 1. + f_A * K_Gli_Oli * glia
    aux_1 /= 1. + K_Gli_Oli * ( glia + glir )
    # Repression by Nkx
    aux_2 = 1./(1. + K_Nkx_Oli * nkx)
    aux_2 *= aux_2
    # Repression by Irx
    aux_3 = 1./(1. + K_Irx_Oli * irx)
    aux_3 *= aux_3
    f_oli = alpha_Oli * Hill(K_Pol_Oli  * aux_1 * aux_2 * aux_3)
    f_oli += - delta * oli

    # Nkx
    # Activation by Gli
    aux_1 = 1. + f_A * K_Gli_Nkx * glia
    aux_1 /= 1. + K_Gli_Nkx * ( glia + glir )
    # Repression by Pax
    aux_2 = 1./(1. + K_Pax_Nkx * pax)
    aux_2 *= aux_2
    # Repression by Olig
    aux_3 = 1./(1. + K_Oli_Nkx * oli)
    aux_3 *= aux_3
    # Repression by Irx
    aux_4 = 1./(1. + K_Irx_Nkx * irx)
    aux_4 *= aux_4
    f_nkx = alpha_Nkx * Hill(K_Pol_Nkx  * aux_1 * aux_2 * aux_3 * aux_4)
    f_nkx += - delta * nkx

    # Irx
    # Repression by Olig
    aux_1 = 1./(1. + K_Oli_Irx * oli)
    aux_1 *= aux_1
    # Repression by Nkx
    aux_2 = 1./(1. + K_Nkx_Irx * nkx)
    aux_2 *= aux_2
    f_irx = alpha_Irx * Hill(K_Pol_Irx  * aux_1 * aux_2)
    f_irx += - delta * irx
        
    return np.array([f_ptc, f_glif, f_glia, f_glir, f_pax, f_oli, f_nkx, f_irx])


class GRN(object):
    """
        GRN object

    """
    
    def __init__(self,state,GRN_function,sig_input,lost_morphogen=None):
        self.state=state #state[i] is the state of the GRN
        self.GRN_function=GRN_function #GRN function for all cells
        self.sig_input=sig_input
        self.lost_morphogen=np.zeros(len(state))

    @property
    def n_cells(self):
        return len(self.state)

    @property
    def diff_rates(self):

        # print(f'time_hours = {time_hours}')
        # print(f'diff_rate_hours = {diff_rate_hours}')
        # print(f'diff_rate pmn = {time_hours * diff_rate_hours * 2.}')
        # print(f'diff_rate oth = {time_hours * diff_rate_hours * .5}')
        # print(1./(5. * time_hours * diff_rate_hours))
        # exit()

        rates = np.zeros(self.n_cells)
        olig_high = np.where(self.state[:,idx['Oli']] > .7)[0]
        rates[olig_high] = 1./15.
        return rates
        
    def evolve(self,time,dt,sig_input,bind_rate): #could put self.sig_input...
        """
            One time step dt from time to time+dt with Euler method
        
        """
        # print(f'\nshape GRN state before = {self.state.shape}')

        self.lost_morphogen = bind_rate * sig_input * self.state[:,idx['Ptc']]
        f = np.array( list(map(self.GRN_function,self.state, np.repeat(time,self.n_cells), sig_input)) )
        self.state = self.state + dt * f
        # print(f'shape GRN state after = {self.state.shape}')
    
    def divide(self,ready,factor=1.):
        '''
            Define gene expression in daughter cells

        '''
        # print(f'cells before divisions: {self.n_cells}')
        # print(f'cells dividing: {len(ready)}')
        for k in ready:
            # gene expression in daughter is 'factor' times the one of the parent
            daughter_state = factor*self.state[k].copy() 
            old_state = self.state.copy()
            aux = np.vstack((old_state,daughter_state))
            self.state = np.vstack((aux, daughter_state))
        # print(f'cells after divisions: {self.n_cells}')

        
def build_GRN(n, GRN_function=shh_gli_poni, state=None , sig_input=None):
    """
        Function to initialize the GRN_basic object
        ..Why this was not included in the __init__ of the GRN, is a mystery..
    """
    if state is None:
        state= np.array([[1.,      1., .00001,  1.,
                          1., .00001, 0.00001,  1.]]*n)
    else: 
        state=state

    if sig_input is None:
        sig_input = np.zeros(n) #no signal to start
    else:
        sig_input = sig_input

    return GRN(state, GRN_function, sig_input) 
